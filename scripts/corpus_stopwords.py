from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

from corpus_records import read_jsonl
from corpus_tokenisation import build_tokenised_corpus, file_sha256, tokenise, write_tokenised_corpus


STOPWORD_PIPELINE_VERSION = "1.0.0"


def resolve_project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_root = config_path.resolve().parent.parent
    return project_root / path


def load_reviewed_stopwords(path: Path, language: str, required_scope: str, tokenisation_config: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("language") != language:
        raise ValueError(f"Stopword language mismatch for {path}")
    if payload.get("scope") != required_scope:
        raise ValueError(f"Stopword scope mismatch for {path}")
    if payload.get("status") != "approved":
        raise ValueError(f"Stopword list is pending for {language}")
    if not payload.get("reviewed_by") or not payload.get("reviewed_at"):
        raise ValueError(f"Stopword review evidence is incomplete for {language}")
    tokens = set()
    seen = set()
    for entry in payload.get("entries", []):
        token = unicodedata.normalize("NFC", entry["token"])
        if token in seen:
            raise ValueError(f"Duplicate stopword entry {token!r}")
        seen.add(token)
        parsed = tokenise(token, tokenisation_config)
        if parsed != [token]:
            raise ValueError(f"Stopword entry does not match tokenisation {token!r}")
        if entry["decision"] == "pending":
            raise ValueError(f"Approved list contains pending decision {token!r}")
        if entry["decision"] == "exclude_from_index":
            tokens.add(token)
    return tokens, payload


def load_stopword_condition(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, set[str]], dict[str, Any]]:
    condition = json.loads(config_path.read_text(encoding="utf-8"))
    if condition.get("config_version") != "1.0.0":
        raise ValueError("Unsupported stopword config_version")
    tokenisation_path = resolve_project_path(config_path, condition["tokenisation_config"])
    tokenisation = json.loads(tokenisation_path.read_text(encoding="utf-8"))
    stopwords = {}
    review_metadata = {}
    for language in ("en", "lus"):
        list_path = resolve_project_path(config_path, condition["stopword_lists"][language])
        stopwords[language], payload = load_reviewed_stopwords(
            list_path,
            language,
            condition["required_scope"],
            tokenisation,
        )
        review_metadata[language] = {
            "path": condition["stopword_lists"][language],
            "sha256": file_sha256(list_path),
            "list_version": payload["list_version"],
            "reviewed_by": payload["reviewed_by"],
            "reviewed_at": payload["reviewed_at"],
            "excluded_token_count": len(stopwords[language]),
        }
    return condition, tokenisation, stopwords, review_metadata


def build_stopword_condition(input_path: Path, selection_path: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    condition, tokenisation, stopwords, review_metadata = load_stopword_condition(config_path)
    records = read_jsonl(input_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result = build_tokenised_corpus(records, selection, tokenisation, stopwords)
    metadata = {
        "pipeline": "stopword_removal",
        "pipeline_version": STOPWORD_PIPELINE_VERSION,
        "input_sha256": file_sha256(input_path),
        "selection_sha256": file_sha256(selection_path),
        "condition_config_sha256": file_sha256(config_path),
        "condition_config": condition,
        "tokenisation_config": tokenisation,
        "stopword_reviews": review_metadata,
    }
    return result, metadata


def write_stopword_condition(output_dir: Path, result: dict[str, Any], metadata: dict[str, Any]) -> None:
    write_tokenised_corpus(output_dir, result, metadata)
