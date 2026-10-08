from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from corpus_records import load_manifest, normalise_record, sha256_text, write_jsonl
from corpus_tokenisation import tokenise


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    expected = {"URL", "domain", "topic", "has_image", "has_hyperlink"}
    if not rows or set(rows[0]) != expected:
        raise ValueError(f"Unexpected FLORES-200 metadata fields in {path}")
    return rows


def record_url(article_url: str, split: str, language: str) -> str:
    parts = urlsplit(article_url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.extend([("flores_language", language), ("flores_split", split)])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def make_record(
    source: dict[str, Any],
    archive_url: str,
    split: str,
    language: str,
    article_url: str,
    rows: list[int],
    sentences: list[str],
    domains: list[str],
    topics: list[str],
    raw_path: Path,
    raw_sha256: str,
    root: Path,
    retrieved_at: str,
) -> dict[str, Any]:
    title = sentences[0]
    body = "\n\n".join(sentences[1:]) if len(sentences) > 1 else sentences[0]
    raw = {
        "document_type": "article",
        "language": language,
        "canonical_url": record_url(article_url, split, language),
        "fetched_url": archive_url,
        "retrieved_at": retrieved_at,
        "title": title,
        "body": body,
        "authors": [],
        "section": domains[0] if len(domains) == 1 else "mixed",
        "tags": ["flores200", "parallel-text", *domains, *topics],
        "raw_reference": f"{raw_path.relative_to(root).as_posix()}#rows={','.join(str(value) for value in rows)}",
        "raw_sha256": raw_sha256,
        "http_status": 200,
        "content_type": "text/plain; charset=utf-8",
        "parser_version": "flores200-url-groups-1.0.0",
    }
    return normalise_record(raw, source)


def build(config_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    root = config_path.resolve().parent.parent
    source = load_manifest(root / "corpus" / "sources" / "source-manifest.json")[config["source_id"]]
    raw_directory = root / config["output_directory"]
    token_config = json.loads((root / "configs" / "tokenisation_only.json").read_text(encoding="utf-8"))
    records = []
    pairs = []
    inputs: dict[str, Any] = {}
    excluded_groups = []
    seen_documents = {"en": set(), "lus": set()}
    for split in config["splits"]:
        metadata_path = raw_directory / f"{split}-metadata.tsv"
        metadata = read_metadata(metadata_path)
        paths = {language: raw_directory / f"{split}-{language}.txt" for language in ("en", "lus")}
        lines = {language: read_lines(path) for language, path in paths.items()}
        if len(metadata) != len(lines["en"]) or len(metadata) != len(lines["lus"]):
            raise ValueError(f"FLORES-200 {split} metadata and language files contain different row counts")
        inputs[split] = {
            "metadata": {"path": metadata_path.relative_to(root).as_posix(), "sha256": file_sha256(metadata_path), "row_count": len(metadata)},
            **{
                language: {"path": paths[language].relative_to(root).as_posix(), "sha256": file_sha256(paths[language]), "row_count": len(lines[language])}
                for language in ("en", "lus")
            },
        }
        groups: OrderedDict[str, list[int]] = OrderedDict()
        for index, row in enumerate(metadata):
            groups.setdefault(row["URL"], []).append(index)
        for article_index, (article_url, indices) in enumerate(groups.items(), 1):
            sentence_groups = {language: [lines[language][index].strip() for index in indices] for language in ("en", "lus")}
            reasons = []
            if any(not sentence for values in sentence_groups.values() for sentence in values):
                reasons.append("empty_aligned_text")
            if not reasons:
                for language in ("en", "lus"):
                    document_text = "\n".join(sentence_groups[language])
                    if not tokenise(document_text, token_config):
                        reasons.append(f"empty_after_tokenisation_{language}")
                    digest = sha256_text(document_text.casefold())
                    if digest in seen_documents[language]:
                        reasons.append(f"duplicate_document_{language}")
            if reasons:
                excluded_groups.append({"split": split, "url": article_url, "rows": [value + 1 for value in indices], "reasons": sorted(set(reasons))})
                continue
            for language in ("en", "lus"):
                seen_documents[language].add(sha256_text("\n".join(sentence_groups[language]).casefold()))
            domains = sorted({metadata[index]["domain"].strip().casefold() for index in indices if metadata[index]["domain"].strip()})
            topics = sorted({metadata[index]["topic"].strip().casefold() for index in indices if metadata[index]["topic"].strip()})
            pair: dict[str, Any] = {
                "split": split,
                "article_index": article_index,
                "article_url": article_url,
                "source_rows": [value + 1 for value in indices],
                "sentence_pair_count": len(indices),
                "domains": domains,
                "topics": topics,
            }
            for language in ("lus", "en"):
                record = make_record(
                    source,
                    config["archive_url"],
                    split,
                    language,
                    article_url,
                    pair["source_rows"],
                    sentence_groups[language],
                    domains,
                    topics,
                    paths[language],
                    inputs[split][language]["sha256"],
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
                pair[language] = record["record_id"]
                pair[f"{language}_word_count"] = len(tokenise(record["title"] + "\n" + record["body"], token_config))
            pairs.append(pair)
    selection = {
        "algorithm": "flores200_source_url_groups_v1",
        "config": {
            "config_version": config["config_version"],
            "source_id": config["source_id"],
            "languages": ["lus", "en"],
            "splits": config["splits"],
            "document_group": "source_url",
        },
        "input": inputs,
        "source_row_count": sum(item["metadata"]["row_count"] for item in inputs.values()),
        "selected_pair_count": len(pairs),
        "selected_record_count": len(records),
        "pairs": pairs,
        "excluded_groups": excluded_groups,
    }
    return records, selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Build article-level English-Mizo documents from FLORES-200.")
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
    print(f"Built {selection['selected_pair_count']} FLORES-200 paired documents")


if __name__ == "__main__":
    main()
