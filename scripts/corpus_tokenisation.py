from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import sparse

from corpus_records import read_jsonl


TOKENISER_VERSION = "1.0.0"
APOSTROPHES = {"'", "’"}
HYPHENS = {"-", "‐", "‑"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_tokenisation_config(config: dict[str, Any]) -> None:
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    if config.get("unicode_form") != "NFC":
        raise ValueError("Only NFC normalisation is supported")
    fields = config.get("indexed_fields")
    if not fields or any(field not in {"title", "body"} for field in fields):
        raise ValueError("indexed_fields must contain title or body")
    if len(fields) != len(set(fields)):
        raise ValueError("indexed_fields contains duplicates")
    if int(config.get("minimum_letter_count", 0)) < 1:
        raise ValueError("minimum_letter_count must be positive")
    if int(config.get("minimum_document_frequency", 0)) < 1:
        raise ValueError("minimum_document_frequency must be positive")
    ratio = float(config.get("maximum_document_frequency_ratio", 0))
    if not 0 < ratio <= 1:
        raise ValueError("maximum_document_frequency_ratio must lie above 0 and at most 1")


def is_letter_or_mark(character: str) -> bool:
    return unicodedata.category(character)[0] in {"L", "M"}


def is_digit(character: str) -> bool:
    return unicodedata.category(character) == "Nd"


def tokenise(text: str, config: dict[str, Any]) -> list[str]:
    validate_tokenisation_config(config)
    text = unicodedata.normalize("NFC", text)
    if config.get("casefold", True):
        text = text.casefold()
    tokens = []
    current: list[str] = []
    length = len(text)
    for index, character in enumerate(text):
        if is_letter_or_mark(character) or is_digit(character):
            current.append(character)
            continue
        keep_apostrophe = (
            config.get("keep_internal_apostrophe", True)
            and character in APOSTROPHES
            and bool(current)
            and is_letter_or_mark(current[-1])
            and index + 1 < length
            and is_letter_or_mark(text[index + 1])
        )
        keep_hyphen = (
            config.get("keep_internal_hyphen", False)
            and character in HYPHENS
            and bool(current)
            and is_letter_or_mark(current[-1])
            and index + 1 < length
            and is_letter_or_mark(text[index + 1])
        )
        if keep_apostrophe:
            current.append("'")
        elif keep_hyphen:
            current.append("-")
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    output = []
    for token in tokens:
        letter_count = sum(is_letter_or_mark(character) for character in token)
        if letter_count:
            if letter_count < int(config["minimum_letter_count"]):
                continue
        else:
            digit_count = sum(is_digit(character) for character in token)
            if not config.get("include_numeric_tokens", False) or digit_count < int(config["minimum_letter_count"]):
                continue
        output.append(token)
    return output


def selected_record_ids(selection: dict[str, Any]) -> list[str]:
    languages = selection["config"]["languages"]
    identifiers = []
    for pair in selection["pairs"]:
        identifiers.extend(pair[language] for language in languages)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Selection contains repeated record identifiers")
    return identifiers


def build_tokenised_corpus(
    records: list[dict[str, Any]],
    selection: dict[str, Any],
    config: dict[str, Any],
    stopwords_by_language: dict[str, set[str]] | None = None,
    stemmers_by_language: dict[str, Callable[[str], str]] | None = None,
) -> dict[str, Any]:
    validate_tokenisation_config(config)
    by_id = {record["record_id"]: record for record in records}
    identifiers = selected_record_ids(selection)
    missing = sorted(set(identifiers) - set(by_id))
    if missing:
        raise ValueError(f"Selection references missing records {missing}")
    documents = []
    document_tokens = []
    document_frequency: Counter[str] = Counter()
    for record_id in identifiers:
        record = by_id[record_id]
        text = "\n\n".join(record[field] for field in config["indexed_fields"])
        tokens = tokenise(text, config)
        original_token_count = len(tokens)
        if stopwords_by_language is not None:
            tokens = [token for token in tokens if token not in stopwords_by_language.get(record["language"], set())]
        changed_stem_count = 0
        if stemmers_by_language is not None:
            stemmer = stemmers_by_language[record["language"]]
            stemmed_tokens = []
            for token in tokens:
                stemmed = stemmer(token)
                if not stemmed:
                    raise ValueError(f"Stemmer returned an empty token for {token!r}")
                changed_stem_count += stemmed != token
                stemmed_tokens.append(stemmed)
            tokens = stemmed_tokens
        unique_tokens = set(tokens)
        document_frequency.update(unique_tokens)
        document_tokens.append(tokens)
        document = {
            "record_id": record_id,
            "language": record["language"],
            "token_count": len(tokens),
            "unique_token_count": len(unique_tokens),
            "token_sequence_sha256": text_sha256("\n".join(tokens)),
        }
        if stopwords_by_language is not None:
            document["removed_stopword_count"] = original_token_count - len(tokens)
        if stemmers_by_language is not None:
            document["changed_stem_count"] = changed_stem_count
        documents.append(document)
    document_count = len(documents)
    minimum_df = int(config["minimum_document_frequency"])
    maximum_df = float(config["maximum_document_frequency_ratio"]) * document_count
    vocabulary = sorted(token for token, count in document_frequency.items() if count >= minimum_df and count <= maximum_df)
    token_index = {token: index for index, token in enumerate(vocabulary)}
    rows = []
    columns = []
    for row, tokens in enumerate(document_tokens):
        for token in sorted(set(tokens)):
            if token in token_index:
                rows.append(row)
                columns.append(token_index[token])
    data = np.ones(len(rows), dtype=np.uint8)
    matrix = sparse.csr_matrix((data, (rows, columns)), shape=(document_count, len(vocabulary)), dtype=np.uint8)
    vocabulary_rows = [
        {"token": token, "document_frequency": int(document_frequency[token])}
        for token in vocabulary
    ]
    return {
        "documents": documents,
        "vocabulary": vocabulary_rows,
        "matrix": matrix,
        "summary": {
            "document_count": document_count,
            "language_document_counts": dict(sorted(Counter(item["language"] for item in documents).items())),
            "vocabulary_size": len(vocabulary),
            "incidence_count": int(matrix.nnz),
            "matrix_density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])) if matrix.shape[0] and matrix.shape[1] else 0.0,
        },
    }


def write_tokenised_corpus(output_dir: Path, result: dict[str, Any], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents.json"
    vocabulary_path = output_dir / "vocabulary.json"
    matrix_path = output_dir / "incidence.npz"
    documents_path.write_text(json.dumps(result["documents"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    vocabulary_path.write_text(json.dumps(result["vocabulary"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sparse.save_npz(matrix_path, result["matrix"], compressed=False)
    metadata = dict(metadata)
    metadata["summary"] = result["summary"]
    metadata["artifacts"] = {
        "documents_sha256": file_sha256(documents_path),
        "vocabulary_sha256": file_sha256(vocabulary_path),
        "incidence_sha256": file_sha256(matrix_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_from_files(input_path: Path, selection_path: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    records = read_jsonl(input_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = build_tokenised_corpus(records, selection, config)
    metadata = {
        "pipeline": "tokenisation_only",
        "tokeniser_version": TOKENISER_VERSION,
        "input_sha256": file_sha256(input_path),
        "selection_sha256": file_sha256(selection_path),
        "config_sha256": file_sha256(config_path),
        "config": config,
    }
    return result, metadata
