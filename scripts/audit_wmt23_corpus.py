from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from build_wmt23_corpus import file_sha256, load_lines, normalise_sentence
from corpus_tokenisation import tokenise


def summary(values: list[float | int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "minimum": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def proportion(count: int, total: int) -> float:
    return count / total if total else 0.0


def audit(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parent.parent
    split = config["production_split"]
    raw_directory = root / config["output_directory"]
    paths = {language: raw_directory / f"{split}-{language}.txt" for language in ("en", "lus")}
    lines = {language: load_lines(path) for language, path in paths.items()}
    if len(lines["en"]) != len(lines["lus"]):
        raise ValueError("Aligned source files contain different row counts")
    token_config = json.loads((root / "configs" / "tokenisation_only.json").read_text(encoding="utf-8"))
    valid = []
    exclusions = {
        "empty_aligned_text": [],
        "empty_after_tokenisation": [],
        "duplicate_aligned_pair": [],
    }
    replacement_character_rows = {"en": [], "lus": []}
    control_character_rows = {"en": [], "lus": []}
    token_counts = {"en": [], "lus": []}
    seen = {"en": set(), "lus": set()}
    duplicate_rows = {"en": 0, "lus": 0}
    seen_pairs = set()
    cross_language_identical = 0
    for row, (english_raw, mizo_raw) in enumerate(zip(lines["en"], lines["lus"]), 1):
        pair = {"en": normalise_sentence(english_raw), "lus": normalise_sentence(mizo_raw)}
        for language in ("en", "lus"):
            text = pair[language]
            if "\ufffd" in text:
                replacement_character_rows[language].append(row)
            if any(unicodedata.category(character) == "Cc" for character in text):
                control_character_rows[language].append(row)
        if not pair["en"] or not pair["lus"]:
            exclusions["empty_aligned_text"].append(row)
            continue
        counts = {language: len(tokenise(pair[language], token_config)) for language in ("en", "lus")}
        if not counts["en"] or not counts["lus"]:
            exclusions["empty_after_tokenisation"].append(row)
            continue
        pair_digest = hashlib.sha256(
            (pair["en"].casefold() + "\n" + pair["lus"].casefold()).encode("utf-8")
        ).hexdigest()
        if pair_digest in seen_pairs:
            exclusions["duplicate_aligned_pair"].append(row)
            continue
        seen_pairs.add(pair_digest)
        for language in ("en", "lus"):
            digest = hashlib.sha256(pair[language].casefold().encode("utf-8")).hexdigest()
            if digest in seen[language]:
                duplicate_rows[language] += 1
            seen[language].add(digest)
            token_counts[language].append(counts[language])
        if pair["en"].casefold() == pair["lus"].casefold():
            cross_language_identical += 1
        valid.append((row, pair, counts))
    row_ratios = [max(item[2].values()) / min(item[2].values()) for item in valid]
    selection_path = root / config["selection_output"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    block_counts = {
        language: [item[f"{language}_word_count"] for item in selection["pairs"]]
        for language in ("en", "lus")
    }
    block_ratios = [
        max(item["en_word_count"], item["lus_word_count"]) /
        min(item["en_word_count"], item["lus_word_count"])
        for item in selection["pairs"]
    ]
    return {
        "status": "audited",
        "source_id": config["source_id"],
        "split": split,
        "config_sha256": file_sha256(config_path),
        "selection_sha256": file_sha256(selection_path),
        "inputs": {
            language: {
                "path": paths[language].relative_to(root).as_posix(),
                "sha256": file_sha256(paths[language]),
            }
            for language in ("en", "lus")
        },
        "source_row_count": len(lines["en"]),
        "valid_row_count": len(valid),
        "exclusions": exclusions,
        "pre_group_excluded_row_count": sum(len(rows) for rows in exclusions.values()),
        "replacement_character_rows": replacement_character_rows,
        "control_character_rows": control_character_rows,
        "duplicate_row_count": duplicate_rows,
        "cross_language_identical_row_count": cross_language_identical,
        "row_token_counts": {language: summary(values) for language, values in token_counts.items()},
        "row_length_ratio": {
            **summary(row_ratios),
            "above_2_count": sum(value > 2 for value in row_ratios),
            "above_2_proportion": proportion(sum(value > 2 for value in row_ratios), len(row_ratios)),
            "above_4_count": sum(value > 4 for value in row_ratios),
            "above_4_proportion": proportion(sum(value > 4 for value in row_ratios), len(row_ratios)),
        },
        "selected_block_count": selection["selected_pair_count"],
        "selected_row_count": sum(item["sentence_pair_count"] for item in selection["pairs"]),
        "incomplete_final_block_count": sum(
            item["reason"] == "incomplete_final_block" for item in selection["excluded_rows"]
        ),
        "block_token_counts": {language: summary(values) for language, values in block_counts.items()},
        "block_length_ratio": {
            **summary(block_ratios),
            "above_2_count": sum(value > 2 for value in block_ratios),
            "above_2_proportion": proportion(sum(value > 2 for value in block_ratios), len(block_ratios)),
        },
    }


def report(result: dict[str, Any]) -> str:
    row_ratio = result["row_length_ratio"]
    block_ratio = result["block_length_ratio"]
    lines = [
        "# WMT23 corpus audit",
        "",
        f"The training files contained {result['source_row_count']:,} aligned rows. "
        f"{result['valid_row_count']:,} unique rows contained tokenised text in both languages. "
        f"{result['pre_group_excluded_row_count']} rows were excluded before grouping.",
        "",
        f"Empty source text excluded {len(result['exclusions']['empty_aligned_text'])} rows. "
        f"Empty token sequences excluded {len(result['exclusions']['empty_after_tokenisation'])} rows. "
        f"Repeated bilingual pairs excluded {len(result['exclusions']['duplicate_aligned_pair'])} rows.",
        "",
        f"The final partial block excluded {result['incomplete_final_block_count']} rows. "
        f"The selected blocks contain {result['selected_row_count']:,} aligned rows.",
        "",
        "| Measure | English | Mizo |",
        "|---|---:|---:|",
    ]
    for label, key in (("Median tokens per row", "median"), ("Mean tokens per row", "mean"), ("Median tokens per block", "median"), ("Mean tokens per block", "mean")):
        source = result["row_token_counts"] if "row" in label else result["block_token_counts"]
        lines.append(f"| {label} | {source['en'][key]:.2f} | {source['lus'][key]:.2f} |")
    lines.extend([
        "",
        f"The median row length ratio was {row_ratio['median']:.3f}. "
        f"{row_ratio['above_2_count']:,} valid rows exceeded a ratio of 2. "
        f"{row_ratio['above_4_count']:,} exceeded a ratio of 4.",
        "",
        f"The selected corpus contained {result['selected_block_count']:,} paired blocks. "
        f"The median block length ratio was {block_ratio['median']:.3f}. "
        f"{block_ratio['above_2_count']:,} blocks exceeded a ratio of 2.",
        "",
        f"Repeated normalised rows occurred {result['duplicate_row_count']['en']:,} times in English and "
        f"{result['duplicate_row_count']['lus']:,} times in Mizo. "
        f"Cross-language text was identical in {result['cross_language_identical_row_count']:,} valid rows.",
        "",
        f"Replacement characters occurred in {len(result['replacement_character_rows']['en'])} English rows and "
        f"{len(result['replacement_character_rows']['lus'])} Mizo rows. "
        f"Control characters occurred in {len(result['control_character_rows']['en'])} English rows and "
        f"{len(result['control_character_rows']['lus'])} Mizo rows.",
        "",
        "All values were generated by `scripts/audit_wmt23_corpus.py`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the WMT23 English-Mizo production corpus.")
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(report(result), encoding="utf-8")
    print(f"Audited {result['valid_row_count']} aligned rows")


if __name__ == "__main__":
    main()
