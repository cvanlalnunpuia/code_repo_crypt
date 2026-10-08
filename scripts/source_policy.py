from __future__ import annotations

from datetime import date
from typing import Any
from urllib.parse import urlsplit


class SourcePolicyError(RuntimeError):
    pass


def validate_source_entry(source: dict[str, Any]) -> list[str]:
    errors = []
    required = {
        "source_id", "name", "source_kind", "base_url", "allowed_hosts",
        "languages", "status", "permission_status", "storage_mode",
        "redistribution_mode", "robots_url", "terms_url", "robots_review",
        "project_authorisation", "permission_evidence", "parser_id", "review_note",
        "minimum_request_interval_seconds",
    }
    missing = sorted(required - set(source))
    if missing:
        errors.append(f"missing fields {missing}")
        return errors
    if source["source_kind"] not in {"publisher", "archive", "synthetic"}:
        errors.append("invalid source_kind")
    if source["status"] not in {"approved", "candidate", "excluded"}:
        errors.append("invalid status")
    if source["permission_status"] not in {"approved", "permission_required", "excluded"}:
        errors.append("invalid permission_status")
    if source["storage_mode"] not in {"full_text", "derived_only", "none"}:
        errors.append("invalid storage_mode")
    if source["redistribution_mode"] not in {"full_text", "metadata_only", "none"}:
        errors.append("invalid redistribution_mode")
    if not source["allowed_hosts"] or len(source["allowed_hosts"]) != len(set(source["allowed_hosts"])):
        errors.append("allowed_hosts must contain unique values")
    base_host = (urlsplit(source["base_url"]).hostname or "").lower()
    if base_host not in source["allowed_hosts"]:
        errors.append("base_url host is absent from allowed_hosts")
    if source["minimum_request_interval_seconds"] < 0:
        errors.append("minimum_request_interval_seconds must be non-negative")
    if source["project_authorisation"] is not None and source["project_authorisation"].get("status") != "approved":
        errors.append("invalid project_authorisation")
    if source["status"] == "approved":
        if not source["project_authorisation"]:
            errors.append("approved source lacks project authorisation")
        if source["permission_status"] != "approved":
            errors.append("approved source lacks approved permission")
        if source["storage_mode"] == "none":
            errors.append("approved source lacks a storage mode")
        if not source["permission_evidence"]:
            errors.append("approved source lacks permission evidence")
        if not source["parser_id"]:
            errors.append("approved source lacks a parser_id")
        if source["source_kind"] == "publisher":
            review = source["robots_review"]
            if not review or not review.get("collection_allowed"):
                errors.append("approved external source lacks an allowed robots review")
    return errors


def assert_acquisition_allowed(source: dict[str, Any], url: str, language: str, parser_ids: set[str]) -> None:
    errors = validate_source_entry(source)
    if errors:
        raise SourcePolicyError(", ".join(errors))
    if source["status"] != "approved":
        raise SourcePolicyError(f"Source {source['source_id']} is disabled")
    if source["permission_status"] != "approved":
        raise SourcePolicyError(f"Source {source['source_id']} lacks permission")
    if source["storage_mode"] != "full_text":
        raise SourcePolicyError(f"Source {source['source_id']} cannot store full text")
    if language not in source["languages"]:
        raise SourcePolicyError(f"Language {language} is outside the source manifest")
    host = (urlsplit(url).hostname or "").lower()
    if host not in source["allowed_hosts"]:
        raise SourcePolicyError(f"Host {host} is outside the source allowlist")
    if source["parser_id"] not in parser_ids:
        raise SourcePolicyError(f"Parser {source['parser_id']} is unavailable")


def parse_review_date(value: str) -> date:
    return date.fromisoformat(value)
