from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SCHEMA_VERSION = "1.0.0"
NORMALISER_VERSION = "1.0.0"
TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalise_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    paragraphs = []
    for block in re.split(r"\n\s*\n", value):
        line = re.sub(r"[\t \f\v]+", " ", block)
        line = re.sub(r" *\n *", " ", line).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def normalise_url(value: str) -> str:
    parts = urlsplit(value.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"Invalid article URL {value!r}")
    host = parts.hostname.lower() if parts.hostname else ""
    port = parts.port
    if port and not ((parts.scheme.lower() == "http" and port == 80) or (parts.scheme.lower() == "https" and port == 443)):
        host = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMETERS
    ]
    return urlunsplit((parts.scheme.lower(), host, path, urlencode(sorted(query)), ""))


def parse_timestamp(value: Any, field: str, required: bool) -> str | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"Missing {field}")
        return None
    text = str(value)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a time-zone offset")
    return parsed.isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["source_id"]: item for item in payload["sources"]}


def normalise_record(raw: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    if source["status"] != "approved" or source["permission_status"] != "approved":
        raise PermissionError(f"Source {source['source_id']} is disabled")
    if source["storage_mode"] != "full_text":
        raise PermissionError(f"Source {source['source_id']} does not permit full-text storage")
    language = raw.get("language")
    if language not in source["languages"] or language not in {"en", "lus"}:
        raise ValueError(f"Unsupported language {language!r}")
    canonical_url = normalise_url(raw.get("canonical_url") or raw["fetched_url"])
    title = normalise_text(str(raw["title"]))
    body = normalise_text(str(raw["body"]))
    if not title or not body:
        raise ValueError("Title and body must contain text")
    authors = sorted({normalise_text(str(item)) for item in raw.get("authors", []) if normalise_text(str(item))})
    tags = sorted({normalise_text(str(item)) for item in raw.get("tags", []) if normalise_text(str(item))})
    record_id = sha256_text(f"{source['source_id']}\n{canonical_url}")
    raw_sha = raw.get("raw_sha256")
    if raw_sha is not None and not re.fullmatch(r"[a-f0-9]{64}", raw_sha):
        raise ValueError("raw_sha256 must contain 64 lowercase hexadecimal characters")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "source_id": source["source_id"],
        "document_type": raw.get("document_type", "article"),
        "language": language,
        "language_confidence": raw.get("language_confidence"),
        "canonical_url": canonical_url,
        "fetched_url": raw.get("fetched_url"),
        "published_at": parse_timestamp(raw.get("published_at"), "published_at", False),
        "updated_at": parse_timestamp(raw.get("updated_at"), "updated_at", False),
        "retrieved_at": parse_timestamp(raw.get("retrieved_at"), "retrieved_at", True),
        "title": title,
        "body": body,
        "authors": authors,
        "section": normalise_text(str(raw["section"])) if raw.get("section") else None,
        "tags": tags,
        "content_sha256": sha256_text(body),
        "normalised_text_sha256": sha256_text(title.casefold() + "\n" + body.casefold()),
        "provenance": {
            "raw_reference": raw.get("raw_reference"),
            "raw_sha256": raw_sha,
            "http_status": raw.get("http_status"),
            "content_type": raw.get("content_type"),
        },
        "rights": {
            "permission_status": source["permission_status"],
            "storage_mode": source["storage_mode"],
            "redistribution_mode": source["redistribution_mode"],
        },
        "processing": {
            "normaliser_version": NORMALISER_VERSION,
            "parser_version": str(raw.get("parser_version", "fixture-1.0.0")),
            "unicode_form": "NFC",
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def token_shingles(record: dict[str, Any], width: int = 3) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\w+", (record["title"] + " " + record["body"]).casefold(), flags=re.UNICODE)
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index:index + width]) for index in range(len(tokens) - width + 1)}


def jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def deduplicate(records: list[dict[str, Any]], threshold: float = 0.85) -> list[dict[str, Any]]:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie from 0 to 1")
    ordered = sorted(records, key=lambda item: (item.get("published_at") or item["retrieved_at"], item["record_id"]))
    representatives: list[dict[str, Any]] = []
    representative_shingles: list[set[tuple[str, ...]]] = []
    output = []
    for record in ordered:
        match = None
        match_status = "unique"
        similarity = 1.0
        current_shingles = token_shingles(record)
        for index, representative in enumerate(representatives):
            if representative["language"] != record["language"]:
                continue
            if representative["normalised_text_sha256"] == record["normalised_text_sha256"]:
                match = representative
                match_status = "exact"
                similarity = 1.0
                break
            score = jaccard(current_shingles, representative_shingles[index])
            if score >= threshold and (match is None or score > similarity):
                match = representative
                match_status = "near"
                similarity = score
        if match is None:
            match = record
            representatives.append(record)
            representative_shingles.append(current_shingles)
        enriched = dict(record)
        enriched["duplicate"] = {
            "status": match_status,
            "canonical_record_id": match["record_id"],
            "group_id": sha256_text(match["record_id"]),
            "similarity": round(similarity, 6),
        }
        output.append(enriched)
    return sorted(output, key=lambda item: item["record_id"])
