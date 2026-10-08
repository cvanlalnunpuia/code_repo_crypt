from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_score(seed: int, pair: dict[str, Any], languages: list[str]) -> str:
    value = "\n".join([str(seed)] + [pair[language] for language in languages])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_split(selection: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    languages = selection["config"]["languages"]
    pairs = selection["pairs"]
    if not pairs:
        raise ValueError("Selection contains no matched pairs")
    ranked = sorted(
        range(len(pairs)),
        key=lambda index: (pair_score(int(config["seed"]), pairs[index], languages), index),
    )
    mode = config["mode"]
    if mode == "shared_fixture_only":
        auxiliary_indices = sorted(ranked)
        client_indices = sorted(ranked)
    elif mode == "disjoint_by_matched_pair":
        if len(pairs) < 2:
            raise ValueError("A disjoint split requires at least two matched pairs")
        fraction = float(config["auxiliary_fraction"])
        if not 0 < fraction < 1:
            raise ValueError("Disjoint auxiliary_fraction must lie between 0 and 1")
        auxiliary_count = int(math.floor(len(pairs) * fraction + 0.5))
        auxiliary_count = max(1, min(len(pairs) - 1, auxiliary_count))
        auxiliary_indices = sorted(ranked[:auxiliary_count])
        client_indices = sorted(ranked[auxiliary_count:])
    else:
        raise ValueError(f"Unknown split mode {mode}")

    def records(indices: list[int], language: str) -> list[str]:
        return [pairs[index][language] for index in indices]

    assignments = {
        language: {
            "auxiliary_record_ids": records(auxiliary_indices, language),
            "client_record_ids": records(client_indices, language),
        }
        for language in languages
    }
    if mode == "disjoint_by_matched_pair":
        for language, assignment in assignments.items():
            overlap = set(assignment["auxiliary_record_ids"]) & set(assignment["client_record_ids"])
            if overlap:
                raise AssertionError(f"Split overlap for {language}")
    return {
        "mode": mode,
        "seed": int(config["seed"]),
        "languages": languages,
        "matched_pair_count": len(pairs),
        "auxiliary_pair_indices": auxiliary_indices,
        "client_pair_indices": client_indices,
        "assignments": assignments,
    }


def build_split_from_files(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.resolve().parent.parent
    selection_path = project_root / config["selection"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result = build_split(selection, config)
    result["config"] = config
    result["config_sha256"] = file_sha256(config_path)
    result["selection_sha256"] = file_sha256(selection_path)
    return result
