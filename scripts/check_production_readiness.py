from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from corpus_records import load_manifest
from source_policy import validate_source_entry
from corpus_stemming import load_stemming_condition


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blocker(code: str, message: str, path: str | None = None) -> dict[str, Any]:
    result = {"code": code, "message": message}
    if path is not None:
        result["path"] = path
    return result


def check_readiness(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    root = config_path.resolve().parent.parent
    blockers = []
    source_manifest_path = root / config["source_manifest"]
    sources = load_manifest(source_manifest_path)
    source = sources.get(config["source_id"])
    if source is None:
        blockers.append(blocker("source_missing", f"Source {config['source_id']} is absent", config["source_manifest"]))
    else:
        entry_errors = validate_source_entry(source)
        for error in entry_errors:
            blockers.append(blocker("source_manifest_invalid", error, config["source_manifest"]))
        if source["status"] != "approved":
            blockers.append(blocker("source_disabled", f"Source status is {source['status']}", config["source_manifest"]))
        if source["permission_status"] != "approved" or not source["permission_evidence"]:
            blockers.append(blocker("source_permission_missing", "Source permission or licence evidence is incomplete", config["source_manifest"]))
        if source["source_kind"] == "publisher":
            review = source["robots_review"]
            if not review or not review.get("collection_allowed"):
                blockers.append(blocker("robots_review_missing", "Robots review does not approve collection", config["source_manifest"]))

    stopword_path = root / config["stopword_config"]
    try:
        stopword_config = json.loads(stopword_path.read_text(encoding="utf-8"))
        for language, relative in stopword_config["stopword_lists"].items():
            list_path = root / relative
            payload = json.loads(list_path.read_text(encoding="utf-8"))
            if payload.get("status") != "approved" or not payload.get("reviewed_by") or not payload.get("reviewed_at"):
                blockers.append(blocker(f"stopwords_pending_{language}", f"Reviewed {language} stopword list is pending", relative))
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        blockers.append(blocker("stopword_configuration_invalid", str(exc), config["stopword_config"]))

    stemming_path = root / config["stemming_config"]
    try:
        stemming_config = json.loads(stemming_path.read_text(encoding="utf-8"))
        for language, relative in stemming_config["stemmer_specifications"].items():
            specification_path = root / relative
            payload = json.loads(specification_path.read_text(encoding="utf-8"))
            if payload.get("status") != "approved" or not payload.get("reviewed_by") or not payload.get("reviewed_at"):
                blockers.append(blocker(f"stemmers_pending_{language}", f"Reviewed {language} stemmer specification is pending", relative))
        if stemming_config.get('mizo_execution_policy'):
            load_stemming_condition(stemming_path)
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        blockers.append(blocker("stemming_configuration_invalid", str(exc), config["stemming_config"]))

    annotation_path = root / config["stemmer_annotations"]
    if annotation_path.exists():
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        if annotations.get("status") != "approved" or not annotations.get("items"):
            blockers.append(blocker("annotations_pending", "Production stemmer annotations are pending or empty", config["stemmer_annotations"]))
    else:
        blockers.append(blocker("annotations_missing", "Production stemmer annotations are absent", config["stemmer_annotations"]))

    artifacts = []
    for relative in config["required_artifacts"]:
        path = root / relative
        exists = path.exists()
        artifacts.append({
            "path": relative,
            "exists": exists,
            "sha256": file_sha256(path) if exists and path.is_file() else None,
        })
        if not exists:
            blockers.append(blocker("artifact_missing", "Required production artifact is absent", relative))
    blockers.sort(key=lambda item: (item["code"], item.get("path", ""), item["message"]))
    return {
        "status": "ready" if not blockers else "blocked",
        "config": config,
        "config_sha256": file_sha256(config_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "required_artifacts": artifacts,
    }


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Production readiness",
        "",
        f"Status: {result['status']}",
        "",
        f"Unmet gates: {result['blocker_count']}",
        "",
        "| Code | Path | Requirement |",
        "|---|---|---|",
    ]
    for item in result["blockers"]:
        lines.append(f"| {item['code']} | {item.get('path', '')} | {item['message']} |")
    if not result["blockers"]:
        lines.append("| ready | | All production gates passed |")
    lines.extend([
        "",
        "This report was generated by `scripts/check_production_readiness.py`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check every gate required for production corpus experiments.")
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = check_readiness(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(build_report(result), encoding="utf-8")
    print(f"Production status is {result['status']} with {result['blocker_count']} unmet gates")


if __name__ == "__main__":
    main()
