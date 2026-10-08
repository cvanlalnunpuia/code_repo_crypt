from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from corpus_records import normalise_record, sha256_text, write_jsonl
from corpus_tokenisation import tokenise
from corpus_records import load_manifest


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalise_sentence(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def load_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()


def make_record(
    source: dict[str, Any],
    archive_url: str,
    split: str,
    language: str,
    group_index: int,
    row_numbers: list[int],
    sentences: list[str],
    raw_path: Path,
    raw_sha256: str,
    root: Path,
    retrieved_at: str,
) -> dict[str, Any]:
    query_url = f"{archive_url}?split={split}&language={language}&group={group_index:05d}"
    line_reference = f"{raw_path.relative_to(root).as_posix()}#rows={','.join(str(value) for value in row_numbers)}"
    raw = {
        "document_type": "parallel_text_block",
        "language": language,
        "canonical_url": query_url,
        "fetched_url": archive_url,
        "retrieved_at": retrieved_at,
        "title": sentences[0],
        "body": "\n\n".join(sentences[1:]) if len(sentences) > 1 else sentences[0],
        "authors": [],
        "section": split,
        "tags": ["parallel-text", "wmt23"],
        "raw_reference": line_reference,
        "raw_sha256": raw_sha256,
        "http_status": 200,
        "content_type": "text/plain; charset=utf-8",
        "parser_version": "wmt23-aligned-blocks-1.0.0",
    }
    return normalise_record(raw, source)


def build(config_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    root = config_path.resolve().parent.parent
    source = load_manifest(root / "corpus" / "sources" / "source-manifest.json")[config["source_id"]]
    split = config["production_split"]
    raw_directory = root / config["output_directory"]
    paths = {language: raw_directory / f"{split}-{language}.txt" for language in ("en", "lus")}
    raw_hashes = {language: file_sha256(path) for language, path in paths.items()}
    lines = {language: load_lines(path) for language, path in paths.items()}
    if len(lines["en"]) != len(lines["lus"]):
        raise ValueError("Aligned source files contain different row counts")
    token_config = json.loads((root / "configs" / "tokenisation_only.json").read_text(encoding="utf-8"))
    valid_rows = []
    excluded_rows = []
    seen_pairs = set()
    for index, (english, mizo) in enumerate(zip(lines["en"], lines["lus"]), 1):
        pair = {"en": normalise_sentence(english), "lus": normalise_sentence(mizo)}
        if not pair["en"] or not pair["lus"]:
            excluded_rows.append({"row": index, "reason": "empty_aligned_text"})
            continue
        counts = {language: len(tokenise(pair[language], token_config)) for language in ("en", "lus")}
        if not counts["en"] or not counts["lus"]:
            excluded_rows.append({"row": index, "reason": "empty_after_tokenisation"})
            continue
        pair_digest = sha256_text(pair["en"].casefold() + "\n" + pair["lus"].casefold())
        if pair_digest in seen_pairs:
            excluded_rows.append({"row": index, "reason": "duplicate_aligned_pair"})
            continue
        seen_pairs.add(pair_digest)
        valid_rows.append((index, pair))
    width = int(config["sentences_per_document"])
    if width <= 0:
        raise ValueError("sentences_per_document must be positive")
    blocks = []
    seen = {"en": set(), "lus": set()}
    duplicate_blocks = []
    for start in range(0, len(valid_rows), width):
        chunk = valid_rows[start:start + width]
        if len(chunk) < width and not config["include_incomplete_document"]:
            for row, _ in chunk:
                excluded_rows.append({"row": row, "reason": "incomplete_final_block"})
            continue
        texts = {language: "\n".join(item[language] for _, item in chunk) for language in ("en", "lus")}
        hashes = {language: sha256_text(texts[language].casefold()) for language in ("en", "lus")}
        repeated = [language for language in ("en", "lus") if hashes[language] in seen[language]]
        if repeated:
            duplicate_blocks.append({"rows": [row for row, _ in chunk], "languages": repeated})
            continue
        for language in ("en", "lus"):
            seen[language].add(hashes[language])
        blocks.append(chunk)
    records = []
    pairs = []
    for group_index, chunk in enumerate(blocks, 1):
        row_numbers = [row for row, _ in chunk]
        pair_entry: dict[str, Any] = {
            "source_rows": row_numbers,
            "sentence_pair_count": len(chunk),
        }
        for language in ("lus", "en"):
            sentences = [item[language] for _, item in chunk]
            record = make_record(
                source,
                config["archive_url"],
                split,
                language,
                group_index,
                row_numbers,
                sentences,
                paths[language],
                raw_hashes[language],
                root,
                config["retrieved_at"],
            )
            record["duplicate"] = {
                "status": "unique",
                "canonical_record_id": record["record_id"],
                "group_id": sha256_text(record["record_id"]),
                "similarity": 1.0,
            }
            records.append(record)
            pair_entry[language] = record["record_id"]
            pair_entry[f"{language}_word_count"] = len(tokenise(record["title"] + "\n" + record["body"], token_config))
        pairs.append(pair_entry)
    selection = {
        "algorithm": "aligned_consecutive_sentence_blocks_v1",
        "config": {
            "config_version": config["config_version"],
            "source_id": config["source_id"],
            "languages": ["lus", "en"],
            "split": split,
            "sentences_per_document": width,
            "include_incomplete_document": config["include_incomplete_document"],
        },
        "input": {
            language: {
                "path": paths[language].relative_to(root).as_posix(),
                "sha256": raw_hashes[language],
                "row_count": len(lines[language]),
            }
            for language in ("en", "lus")
        },
        "source_row_count": len(lines["en"]),
        "valid_aligned_row_count": len(valid_rows),
        "selected_pair_count": len(pairs),
        "selected_record_count": len(records),
        "pairs": pairs,
        "excluded_rows": sorted(excluded_rows, key=lambda item: item["row"]),
        "excluded_duplicate_blocks": duplicate_blocks,
    }
    return records, selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired documents from the WMT23 English-Mizo corpus.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent.parent
    records, selection = build(args.config)
    corpus_output = root / config["corpus_output"]
    selection_output = root / config["selection_output"]
    write_jsonl(corpus_output, records)
    selection["corpus_sha256"] = file_sha256(corpus_output)
    selection["config_sha256"] = file_sha256(args.config)
    selection_output.parent.mkdir(parents=True, exist_ok=True)
    selection_output.write_text(json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Built {selection['selected_pair_count']} paired documents")


if __name__ == "__main__":
    main()
