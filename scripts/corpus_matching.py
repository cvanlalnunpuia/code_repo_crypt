from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_time(value: str) -> datetime:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None:
        raise ValueError("Publication time lacks a time-zone offset")
    return parsed


def word_count(record: dict[str, Any]) -> int:
    return len(re.findall(r"\w+", record["body"], flags=re.UNICODE))


def stable_tie(seed: int, left_id: str, right_id: str) -> str:
    return hashlib.sha256(f"{seed}\n{left_id}\n{right_id}".encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    languages = config.get("languages")
    if not isinstance(languages, list) or len(languages) != 2 or len(set(languages)) != 2:
        raise ValueError("Exactly two distinct languages are required")
    if any(language not in {"en", "lus"} for language in languages):
        raise ValueError("Unsupported language")
    if int(config.get("target_pairs", 0)) <= 0:
        raise ValueError("target_pairs must be positive")
    if float(config.get("max_date_gap_days", -1)) < 0:
        raise ValueError("max_date_gap_days must be non-negative")
    if float(config.get("max_length_ratio", 0)) < 1:
        raise ValueError("max_length_ratio must be at least 1")


def select_matched_pairs(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    left_language, right_language = config["languages"]
    excluded = []
    eligible: dict[str, list[dict[str, Any]]] = {left_language: [], right_language: []}
    for record in sorted(records, key=lambda item: item["record_id"]):
        reason = None
        if record.get("source_id") != config["source_id"]:
            reason = "source"
        elif record.get("language") not in eligible:
            reason = "language"
        elif config.get("require_unique", True) and record.get("duplicate", {}).get("status") != "unique":
            reason = "duplicate"
        elif config.get("require_publication_time", True) and not record.get("published_at"):
            reason = "missing_publication_time"
        elif word_count(record) == 0:
            reason = "empty_body"
        if reason:
            excluded.append({"record_id": record["record_id"], "reason": reason})
        else:
            eligible[record["language"]].append(record)

    max_days = float(config["max_date_gap_days"])
    max_ratio = float(config["max_length_ratio"])
    edges = []
    admissible_counts: dict[str, int] = {record["record_id"]: 0 for language in eligible.values() for record in language}
    for left in eligible[left_language]:
        left_words = word_count(left)
        left_time = parse_time(left["published_at"])
        for right in eligible[right_language]:
            right_words = word_count(right)
            right_time = parse_time(right["published_at"])
            date_gap = abs((left_time - right_time).total_seconds()) / 86400
            length_ratio = max(left_words, right_words) / min(left_words, right_words)
            if date_gap > max_days or length_ratio > max_ratio:
                continue
            date_cost = date_gap / max_days if max_days else 0.0
            length_cost = abs(math.log(length_ratio)) / math.log(max_ratio) if max_ratio > 1 else 0.0
            cost = date_cost + length_cost
            admissible_counts[left["record_id"]] += 1
            admissible_counts[right["record_id"]] += 1
            edges.append((
                cost,
                stable_tie(int(config["tie_break_seed"]), left["record_id"], right["record_id"]),
                left,
                right,
                date_gap,
                length_ratio,
                left_words,
                right_words,
            ))
    edges.sort(key=lambda item: (item[0], item[1]))
    selected_left = set()
    selected_right = set()
    pairs = []
    for cost, _, left, right, date_gap, length_ratio, left_words, right_words in edges:
        if len(pairs) >= int(config["target_pairs"]):
            break
        if left["record_id"] in selected_left or right["record_id"] in selected_right:
            continue
        selected_left.add(left["record_id"])
        selected_right.add(right["record_id"])
        pairs.append({
            left_language: left["record_id"],
            right_language: right["record_id"],
            f"{left_language}_word_count": left_words,
            f"{right_language}_word_count": right_words,
            "date_gap_days": round(date_gap, 6),
            "length_ratio": round(length_ratio, 6),
            "matching_cost": round(cost, 6),
        })

    for language, language_records in eligible.items():
        selected = selected_left if language == left_language else selected_right
        for record in language_records:
            if record["record_id"] not in selected:
                reason = "no_admissible_counterpart" if admissible_counts[record["record_id"]] == 0 else "unselected"
                excluded.append({"record_id": record["record_id"], "reason": reason})
    excluded.sort(key=lambda item: item["record_id"])
    return {
        "config": config,
        "eligible_counts": {language: len(items) for language, items in eligible.items()},
        "selected_pair_count": len(pairs),
        "selected_record_count": len(pairs) * 2,
        "pairs": pairs,
        "exclusions": excluded,
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_selection(input_path: Path, config_path: Path) -> dict[str, Any]:
    from corpus_records import read_jsonl

    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = select_matched_pairs(read_jsonl(input_path), config)
    result["input_sha256"] = file_sha256(input_path)
    result["config_sha256"] = file_sha256(config_path)
    result["algorithm"] = "deterministic_greedy_pair_matching_v1"
    return result
