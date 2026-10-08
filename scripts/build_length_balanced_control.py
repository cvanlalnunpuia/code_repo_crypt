from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_id(flores_pair: dict[str, Any], wmt_pair: dict[str, Any]) -> str:
    value = "\n".join([flores_pair["en"], flores_pair["lus"], wmt_pair["en"], wmt_pair["lus"]])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_score(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}\n{identifier}".encode("utf-8")).hexdigest()


def length_vector(pair: dict[str, Any]) -> tuple[int, int]:
    return int(pair["en_word_count"]), int(pair["lus_word_count"])


def build_matches(flores_pairs: list[dict[str, Any]], wmt_pairs: list[dict[str, Any]], maximum_ratio: float) -> list[dict[str, Any]]:
    if maximum_ratio <= 1:
        raise ValueError("maximum_length_ratio must exceed one")
    flores_lengths = np.asarray([length_vector(pair) for pair in flores_pairs], dtype=float)
    wmt_lengths = np.asarray([length_vector(pair) for pair in wmt_pairs], dtype=float)
    log_differences = np.abs(np.log(flores_lengths[:, None, :] / wmt_lengths[None, :, :]))
    eligible = np.max(log_differences, axis=2) <= math.log(maximum_ratio) + 1e-12
    real_cost = np.sum(log_differences, axis=2)
    row_count, real_column_count = eligible.shape
    dummy_cost = 1000.0
    invalid_cost = 1_000_000.0
    costs = np.full((row_count, real_column_count + row_count), dummy_cost, dtype=float)
    costs[:, :real_column_count] = np.where(eligible, real_cost, invalid_cost)
    rows, columns = linear_sum_assignment(costs)
    matches = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column >= real_column_count or not eligible[row, column]:
            continue
        flores_pair = flores_pairs[row]
        wmt_pair = wmt_pairs[column]
        flores_en, flores_lus = length_vector(flores_pair)
        wmt_en, wmt_lus = length_vector(wmt_pair)
        identifier = match_id(flores_pair, wmt_pair)
        matches.append({
            "match_id": identifier,
            "flores_pair_index": row,
            "wmt23_pair_index": column,
            "flores": {"en": flores_pair["en"], "lus": flores_pair["lus"], "en_tokens": flores_en, "lus_tokens": flores_lus},
            "wmt23": {"en": wmt_pair["en"], "lus": wmt_pair["lus"], "en_tokens": wmt_en, "lus_tokens": wmt_lus},
            "length_ratios": {
                "en": max(flores_en, wmt_en) / min(flores_en, wmt_en),
                "lus": max(flores_lus, wmt_lus) / min(flores_lus, wmt_lus),
            },
            "log_length_distance": float(real_cost[row, column]),
        })
    return sorted(matches, key=lambda item: item["match_id"])


def selection(
    source: dict[str, Any],
    matches: list[dict[str, Any]],
    side: str,
    input_path: str,
    input_sha256: str,
    maximum_ratio: float,
) -> dict[str, Any]:
    pairs = []
    for index, match in enumerate(matches):
        source_pair = source["pairs"][match[f"{side}_pair_index"]]
        item = dict(source_pair)
        item["length_match_id"] = match["match_id"]
        item["length_match_index"] = index
        item["counterpart_pair_index"] = match["wmt23_pair_index" if side == "flores" else "flores_pair_index"]
        pairs.append(item)
    return {
        "algorithm": "joint_bilingual_length_caliper_assignment_v1",
        "config": {
            "config_version": "1.0.0",
            "source_id": source["config"]["source_id"],
            "languages": source["config"]["languages"],
            "maximum_length_ratio": maximum_ratio,
            "matching_side": side,
        },
        "input_selection": input_path,
        "input_selection_sha256": input_sha256,
        "input_pair_count": len(source["pairs"]),
        "selected_pair_count": len(pairs),
        "selected_record_count": len(pairs) * len(source["config"]["languages"]),
        "pairs": pairs,
    }


def linked_split(
    selection_payload: dict[str, Any],
    matches: list[dict[str, Any]],
    auxiliary_indices: list[int],
    client_indices: list[int],
    seed: int,
    fraction: float,
    selection_path: str,
    selection_sha256: str,
) -> dict[str, Any]:
    languages = selection_payload["config"]["languages"]
    pairs = selection_payload["pairs"]
    assignments = {
        language: {
            "auxiliary_record_ids": [pairs[index][language] for index in auxiliary_indices],
            "client_record_ids": [pairs[index][language] for index in client_indices],
        }
        for language in languages
    }
    return {
        "mode": "linked_disjoint_by_length_match",
        "seed": seed,
        "languages": languages,
        "matched_pair_count": len(matches),
        "auxiliary_pair_indices": auxiliary_indices,
        "client_pair_indices": client_indices,
        "assignments": assignments,
        "config": {
            "config_version": "1.0.0",
            "selection": selection_path,
            "mode": "linked_disjoint_by_length_match",
            "auxiliary_fraction": fraction,
            "seed": seed,
        },
        "selection_sha256": selection_sha256,
    }


def build(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    root = config_path.resolve().parent.parent
    flores_path = root / config["flores_selection"]
    wmt_path = root / config["wmt23_selection"]
    flores = json.loads(flores_path.read_text(encoding="utf-8"))
    wmt = json.loads(wmt_path.read_text(encoding="utf-8"))
    if flores["config"]["languages"] != wmt["config"]["languages"]:
        raise ValueError("Corpus language orders differ")
    maximum_ratio = float(config["maximum_length_ratio"])
    matches = build_matches(flores["pairs"], wmt["pairs"], maximum_ratio)
    if len(matches) < 2:
        raise ValueError("Length caliper produced fewer than two matches")
    ratios = {
        language: [match["length_ratios"][language] for match in matches]
        for language in ("en", "lus")
    }
    matching = {
        "algorithm": "maximum_cardinality_then_minimum_log_length_distance_v1",
        "maximum_length_ratio": maximum_ratio,
        "scipy_version": scipy.__version__,
        "flores_input_pair_count": len(flores["pairs"]),
        "wmt23_input_pair_count": len(wmt["pairs"]),
        "matched_pair_count": len(matches),
        "ratio_summary": {
            language: {
                "minimum": min(values),
                "median": statistics.median(values),
                "mean": statistics.mean(values),
                "maximum": max(values),
            }
            for language, values in ratios.items()
        },
        "inputs": {
            "flores": {"path": config["flores_selection"], "sha256": file_sha256(flores_path)},
            "wmt23": {"path": config["wmt23_selection"], "sha256": file_sha256(wmt_path)},
        },
        "matches": matches,
    }
    selections = {
        "flores": selection(flores, matches, "flores", config["flores_selection"], file_sha256(flores_path), maximum_ratio),
        "wmt23": selection(wmt, matches, "wmt23", config["wmt23_selection"], file_sha256(wmt_path), maximum_ratio),
    }
    ranked = sorted(range(len(matches)), key=lambda index: (split_score(int(config["split_seed"]), matches[index]["match_id"]), index))
    fraction = float(config["auxiliary_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("auxiliary_fraction must lie between zero and one")
    auxiliary_count = max(1, min(len(matches) - 1, int(math.floor(len(matches) * fraction + 0.5))))
    auxiliary_indices = sorted(ranked[:auxiliary_count])
    client_indices = sorted(ranked[auxiliary_count:])
    return {
        "matching": matching,
        "selections": selections,
        "split_indices": {"auxiliary": auxiliary_indices, "client": client_indices},
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build linked FLORES-200 and WMT23 selections balanced on bilingual token length.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent.parent
    result = build(args.config)
    matching = result["matching"]
    matching["config_sha256"] = file_sha256(args.config)
    write_json(root / config["matching_output"], matching)
    selection_paths = {
        "flores": root / config["flores_output_selection"],
        "wmt23": root / config["wmt23_output_selection"],
    }
    split_paths = {
        "flores": root / config["flores_output_split"],
        "wmt23": root / config["wmt23_output_split"],
    }
    for side in ("flores", "wmt23"):
        selection_payload = result["selections"][side]
        write_json(selection_paths[side], selection_payload)
        split_payload = linked_split(
            selection_payload,
            matching["matches"],
            result["split_indices"]["auxiliary"],
            result["split_indices"]["client"],
            int(config["split_seed"]),
            float(config["auxiliary_fraction"]),
            selection_paths[side].relative_to(root).as_posix(),
            file_sha256(selection_paths[side]),
        )
        write_json(split_paths[side], split_payload)
    print(f"Built {matching['matched_pair_count']} linked length-balanced pairs")


if __name__ == "__main__":
    main()
