from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

from corpus_records import read_jsonl, sha256_text


HEX64 = re.compile(r"^[a-f0-9]{64}$")
REQUIRED = {
    "schema_version", "record_id", "source_id", "document_type", "language",
    "canonical_url", "retrieved_at", "title", "body", "content_sha256",
    "normalised_text_sha256", "provenance", "rights", "processing",
}


def validate_record(record: dict, require_duplicate: bool) -> list[str]:
    errors = []
    missing = sorted(REQUIRED - set(record))
    if missing:
        errors.append(f"missing fields {missing}")
    if record.get("schema_version") != "1.0.0":
        errors.append("unsupported schema_version")
    if record.get("document_type") not in {"article", "parallel_text_block"}:
        errors.append("unsupported document_type")
    if record.get("language") not in {"en", "lus"}:
        errors.append("language must be en or lus")
    for field in ("record_id", "content_sha256", "normalised_text_sha256"):
        if not HEX64.fullmatch(str(record.get(field, ""))):
            errors.append(f"invalid {field}")
    if record.get("content_sha256") != sha256_text(record.get("body", "")):
        errors.append("content_sha256 does not match body")
    if unicodedata.normalize("NFC", record.get("title", "")) != record.get("title"):
        errors.append("title is not NFC")
    if unicodedata.normalize("NFC", record.get("body", "")) != record.get("body"):
        errors.append("body is not NFC")
    parts = urlsplit(record.get("canonical_url", ""))
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.fragment:
        errors.append("invalid canonical_url")
    rights = record.get("rights", {})
    if rights.get("permission_status") != "approved" or rights.get("storage_mode") != "full_text":
        errors.append("full-text record lacks approved storage rights")
    if require_duplicate and "duplicate" not in record:
        errors.append("missing duplicate annotation")
    if "duplicate" in record:
        duplicate = record["duplicate"]
        if duplicate.get("status") not in {"unique", "exact", "near"}:
            errors.append("invalid duplicate status")
        if not HEX64.fullmatch(str(duplicate.get("canonical_record_id", ""))):
            errors.append("invalid duplicate canonical_record_id")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate normalised corpus JSONL.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--require-duplicate", action="store_true")
    args = parser.parse_args()
    json.loads(args.schema.read_text(encoding="utf-8"))
    records = read_jsonl(args.input)
    all_errors = []
    identifiers = set()
    for index, record in enumerate(records, 1):
        errors = validate_record(record, args.require_duplicate)
        if record.get("record_id") in identifiers:
            errors.append("duplicate record_id")
        identifiers.add(record.get("record_id"))
        all_errors.extend(f"line {index}: {error}" for error in errors)
    if all_errors:
        raise SystemExit("\n".join(all_errors))
    print(f"Validated {len(records)} records")


if __name__ == "__main__":
    main()
