from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from analyse_matched_domain_comparison import file_sha256, holm_adjust, marker_summary, paired_summary, run_index
from corpus_records import read_jsonl


METRICS = ["query_weighted_recovery", "distinct_keyword_macro_recovery"]


def keyword_size(dataset_id: str) -> int:
    match = re.search(r":k([0-9]+)$", dataset_id)
    if not match:
        raise ValueError(f"Dataset lacks a keyword-size suffix: {dataset_id}")
    return int(match.group(1))


def build(root: Path) -> dict[str, Any]:
    flores_path = root / "results" / "corpus" / "flores200-length-balanced-experiments.json"
    wmt_path = root / "results" / "corpus" / "wmt23-length-balanced-experiments.json"
    matching_path = root / "results" / "corpus" / "flores200-wmt23-length-matches.json"
    previous_path = root / "results" / "corpus" / "keyword-size-sensitivity-comparison.json"
    flores_selection_path = root / "results" / "corpus" / "flores200-length-balanced-selection.json"
    wmt_selection_path = root / "results" / "corpus" / "wmt23-length-balanced-selection.json"
    flores = json.loads(flores_path.read_text(encoding="utf-8"))
    wmt = json.loads(wmt_path.read_text(encoding="utf-8"))
    matching = json.loads(matching_path.read_text(encoding="utf-8"))
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    for field in ("seeds", "query_distributions", "iteration_checkpoints"):
        if flores["config"][field] != wmt["config"][field]:
            raise ValueError(f"Length-balanced experiment field differs: {field}")
    flores_runs = run_index(flores)
    wmt_runs = run_index(wmt)
    if set(flores_runs) != set(wmt_runs):
        raise ValueError("Length-balanced run keys differ")
    final_checkpoint = max(flores["config"]["iteration_checkpoints"])
    checkpoint_index = flores["config"]["iteration_checkpoints"].index(final_checkpoint)
    comparisons = []
    datasets = sorted({key[0] for key in flores_runs}, key=lambda value: (keyword_size(value), value))
    for dataset in datasets:
        size = keyword_size(dataset)
        language = dataset.split(":", 1)[0]
        for distribution in flores["config"]["query_distributions"]:
            keys = [(dataset, distribution, int(seed)) for seed in flores["config"]["seeds"]]
            for metric in METRICS:
                flores_values = [flores_runs[key]["scores"][checkpoint_index][metric] for key in keys]
                wmt_values = [wmt_runs[key]["scores"][checkpoint_index][metric] for key in keys]
                item = paired_summary(flores_values, wmt_values)
                item.update({
                    "keyword_count": size,
                    "language": language,
                    "dataset_id": dataset,
                    "query_distribution": distribution,
                    "metric": metric,
                    "checkpoint": final_checkpoint,
                    "seeds": [int(seed) for seed in flores["config"]["seeds"]],
                })
                comparisons.append(item)
    holm_adjust(comparisons)
    focus = [
        item for item in comparisons
        if item["language"] == "en"
        and item["query_distribution"] == "zipf_iid"
        and item["metric"] == "query_weighted_recovery"
    ]
    focus.sort(key=lambda item: item["keyword_count"])
    previous_focus = {
        int(item["keyword_count"]): item
        for item in previous["english_zipf_query_weighted_by_size"]
    }
    focus_comparison = [
        {
            "keyword_count": item["keyword_count"],
            "matched_size_difference": previous_focus[item["keyword_count"]]["mean_difference_flores_minus_wmt23"],
            "matched_size_holm_adjusted_p": previous_focus[item["keyword_count"]]["holm_adjusted_p"],
            "length_balanced_difference": item["mean_difference_flores_minus_wmt23"],
            "length_balanced_95_percent_confidence_interval": item["difference_95_percent_confidence_interval"],
            "length_balanced_exact_sign_flip_p": item["exact_sign_flip_p"],
            "length_balanced_holm_adjusted_p": item["holm_adjusted_p"],
        }
        for item in focus
    ]
    token_config = json.loads((root / "configs" / "tokenisation_only.json").read_text(encoding="utf-8"))
    flores_selection = json.loads(flores_selection_path.read_text(encoding="utf-8"))
    wmt_selection = json.loads(wmt_selection_path.read_text(encoding="utf-8"))
    flores_records_path = root / "results" / "corpus" / "flores200-records.jsonl"
    wmt_records_path = root / "results" / "corpus" / "wmt23-records.jsonl"
    return {
        "status": "analysed",
        "checkpoint": final_checkpoint,
        "holm_family_size": len(comparisons),
        "matched_pair_count": matching["matched_pair_count"],
        "maximum_length_ratio": matching["maximum_length_ratio"],
        "length_ratio_summary": matching["ratio_summary"],
        "comparisons": comparisons,
        "english_zipf_query_weighted_comparison": focus_comparison,
        "religious_markers": {
            "flores200": marker_summary(read_jsonl(flores_records_path), flores_selection, token_config),
            "wmt23": marker_summary(read_jsonl(wmt_records_path), wmt_selection, token_config),
        },
        "inputs": {
            "flores_experiment": {"path": flores_path.relative_to(root).as_posix(), "sha256": file_sha256(flores_path)},
            "wmt23_experiment": {"path": wmt_path.relative_to(root).as_posix(), "sha256": file_sha256(wmt_path)},
            "matching": {"path": matching_path.relative_to(root).as_posix(), "sha256": file_sha256(matching_path)},
            "previous_sensitivity": {"path": previous_path.relative_to(root).as_posix(), "sha256": file_sha256(previous_path)},
        },
    }


def interval(value: list[float]) -> str:
    return f"[{value[0]:.4f}, {value[1]:.4f}]"


def markdown(result: dict[str, Any]) -> str:
    focus_rows = []
    for item in result["english_zipf_query_weighted_comparison"]:
        focus_rows.append(
            f"| {item['keyword_count']} | {item['matched_size_difference']:.4f} | "
            f"{item['length_balanced_difference']:.4f} | "
            f"{interval(item['length_balanced_95_percent_confidence_interval'])} | "
            f"{item['length_balanced_exact_sign_flip_p']:.4f} | "
            f"{item['length_balanced_holm_adjusted_p']:.4f} |"
        )
    rows = []
    for item in result["comparisons"]:
        language = {"en": "English", "lus": "Mizo"}[item["language"]]
        distribution = {"uniform_iid": "uniform IID", "zipf_iid": "Zipf IID"}[item["query_distribution"]]
        metric = "query-weighted" if item["metric"] == "query_weighted_recovery" else "keyword-macro"
        rows.append(
            f"| {item['keyword_count']} | {language} | {distribution} | {metric} | "
            f"{item['flores_mean']:.4f} | {item['wmt23_mean']:.4f} | "
            f"{item['mean_difference_flores_minus_wmt23']:.4f} | "
            f"{interval(item['difference_95_percent_confidence_interval'])} | "
            f"{item['holm_adjusted_p']:.4f} |"
        )
    significant = [item for item in result["comparisons"] if item["holm_adjusted_p"] < 0.05]
    language_counts = {
        language: sum(item["language"] == language for item in significant)
        for language in ("en", "lus")
    }
    marker_rows = []
    for corpus, languages in result["religious_markers"].items():
        for language in ("en", "lus"):
            item = languages[language]
            language_label = {"en": "English", "lus": "Mizo"}[language]
            marker_rows.append(
                f"| {corpus} | {language_label} | "
                f"{item['any_marker_document_count']} of {item['document_count']} | "
                f"{item['any_marker_document_percentage']:.3f}% |"
            )
    return f"""# Length-balanced corpus comparison

The control retained {result['matched_pair_count']} one-to-one corpus pairs. English and Mizo token-count ratios were limited to {result['maximum_length_ratio']:.2f} within every cross-corpus match. Linked splitting assigned corresponding pairs to the same 84-document auxiliary or 169-document client partition.

## English Zipf recovery

Differences are FLORES-200 minus WMT23 at 1,000 IHOP iterations.

| Keywords | 562-pair difference | Length-balanced difference | 95% CI | Exact p | Holm p |
|---:|---:|---:|---|---:|---:|
{chr(10).join(focus_rows)}

## Recovery

Holm adjustment covered all {result['holm_family_size']} length-balanced comparisons.

| Keywords | Language | Queries | Metric | FLORES-200 | WMT23 | Difference | 95% CI | Holm p |
|---:|---|---|---|---:|---:|---:|---|---:|
{chr(10).join(rows)}

{len(significant)} comparisons had Holm-adjusted p below 0.05. The counts were {language_counts['en']} for English and {language_counts['lus']} for Mizo.

## Religious markers

| Corpus | Language | Documents with marker | Percentage |
|---|---|---:|---:|
{chr(10).join(marker_rows)}

Length balance reduced the corpus to its region of common support. Source domain, article construction, and lexical composition still differed. The comparison estimates the association with corpus choice after controlling document count, linked partition assignment, bilingual token length, keyword count, seeds, and query generation.

All values were generated by `scripts/analyse_length_balanced_comparison.py`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse the linked length-balanced corpus comparison.")
    parser.add_argument("--output", type=Path, default=Path("results/corpus/length-balanced-comparison.json"))
    parser.add_argument("--document", type=Path, default=Path("docs/length-balanced-comparison.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    result = build(root)
    output = root / args.output
    document = root / args.document
    output.parent.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    document.write_text(markdown(result), encoding="utf-8")
    print(f"Analysed {len(result['comparisons'])} length-balanced comparisons")


if __name__ == "__main__":
    main()
