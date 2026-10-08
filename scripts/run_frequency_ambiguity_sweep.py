from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "ihop-code"
CONFIG_PATH = ROOT / "configs" / "frequency_ambiguity_sweep.json"
DATASET = UPSTREAM / "datasets_pro" / "enron-full.pkl"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT / "scripts"))

from defense import generate_observations
from experiment import generate_keyword_queries, generate_train_test_data, run_attack
from measure_seed_structure import canonical_hash, profile_matrix, restricted_matrix
from run_access_similarity_sweep import (
    accuracy,
    build_params,
    file_sha256,
    frequency_vector,
    git_revision,
    pair_jaccards,
    sparse_matrix_sha256,
    summary,
)


def select_pair_pool(
    selected_keywords: list[int],
    client_frequency: np.ndarray,
    auxiliary_frequency: np.ndarray,
    pair_count: int,
) -> list[dict]:
    candidates = []
    keyword_count = len(selected_keywords)
    for first in range(keyword_count):
        for second in range(first + 1, keyword_count):
            client_first = int(client_frequency[first])
            client_second = int(client_frequency[second])
            auxiliary_first = int(auxiliary_frequency[first])
            auxiliary_second = int(auxiliary_frequency[second])
            if client_first == client_second or auxiliary_first == auxiliary_second:
                continue
            if (client_first + client_second) % 2 != 0:
                continue
            if (auxiliary_first + auxiliary_second) % 2 != 0:
                continue
            if min(client_first, client_second, auxiliary_first, auxiliary_second) == 0:
                continue
            client_scale = max(client_first, client_second)
            auxiliary_scale = max(auxiliary_first, auxiliary_second)
            score = abs(client_first - client_second) / client_scale
            score += abs(auxiliary_first - auxiliary_second) / auxiliary_scale
            candidates.append((score, first, second))

    candidates.sort()
    selected = []
    used = set()
    for score, first, second in candidates:
        if first in used or second in used:
            continue
        used.update((first, second))
        selected.append(
            {
                "first_position": first,
                "second_position": second,
                "first_keyword_id": int(selected_keywords[first]),
                "second_keyword_id": int(selected_keywords[second]),
                "selection_score": score,
                "client_first_frequency": int(client_frequency[first]),
                "client_second_frequency": int(client_frequency[second]),
                "client_equalized_frequency": (int(client_frequency[first]) + int(client_frequency[second])) // 2,
                "auxiliary_first_frequency": int(auxiliary_frequency[first]),
                "auxiliary_second_frequency": int(auxiliary_frequency[second]),
                "auxiliary_equalized_frequency": (
                    int(auxiliary_frequency[first]) + int(auxiliary_frequency[second])
                )
                // 2,
            }
        )
        if len(selected) == pair_count:
            return selected
    raise RuntimeError(f"Only {len(selected)} eligible disjoint pairs were found")


def local_seed(seed: int, matrix_name: str, first: int, second: int) -> int:
    payload = f"{seed}:{matrix_name}:{first}:{second}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def equalize_pairs(
    dataset: list,
    pairs: list[dict],
    seed: int,
    matrix_name: str,
) -> list[list[int]]:
    documents = [set(document) for document in dataset]
    document_count = len(documents)
    frequency_key = f"{matrix_name}_equalized_frequency"
    for pair in pairs:
        equalized_frequency = pair[frequency_key]
        if 2 * equalized_frequency > document_count:
            raise RuntimeError("Equalized pair cannot be assigned disjoint documents")
        generator = np.random.RandomState(
            local_seed(
                seed,
                matrix_name,
                pair["first_keyword_id"],
                pair["second_keyword_id"],
            )
        )
        permutation = generator.permutation(document_count)
        first_documents = permutation[:equalized_frequency]
        second_documents = permutation[
            equalized_frequency : 2 * equalized_frequency
        ]
        first_keyword = pair["first_keyword_id"]
        second_keyword = pair["second_keyword_id"]
        for document in documents:
            document.discard(first_keyword)
            document.discard(second_keyword)
        for document_index in first_documents:
            documents[int(document_index)].add(first_keyword)
        for document_index in second_documents:
            documents[int(document_index)].add(second_keyword)
    return [sorted(document) for document in documents]


def prepare_base(config: dict, seed: int):
    params = build_params(config)
    np.random.seed(seed)
    previous_cwd = Path.cwd()
    try:
        os.chdir(UPSTREAM)
        auxiliary, client, real_frequencies = generate_train_test_data(
            params.gen_params
        )
    finally:
        os.chdir(previous_cwd)
    selected_keywords = [int(value) for value in client["keywords"]]
    client_matrix = restricted_matrix(client["dataset"], selected_keywords)
    auxiliary_matrix = restricted_matrix(auxiliary["dataset"], selected_keywords)
    pair_pool = select_pair_pool(
        selected_keywords,
        frequency_vector(client_matrix),
        frequency_vector(auxiliary_matrix),
        max(config["equalized_pair_counts"]),
    )
    return (
        params,
        auxiliary,
        client,
        real_frequencies,
        selected_keywords,
        client_matrix,
        auxiliary_matrix,
        pair_pool,
    )


def run_intervention(
    config: dict, structure_config: dict, pair_count: int, seed: int
) -> dict:
    started = datetime.now(timezone.utc)
    tick = time.perf_counter()
    (
        params,
        auxiliary,
        client,
        real_frequencies,
        selected_keywords,
        base_client_matrix,
        base_auxiliary_matrix,
        pair_pool,
    ) = prepare_base(config, seed)
    active_pairs = pair_pool[:pair_count]
    client["dataset"] = equalize_pairs(
        client["dataset"], active_pairs, seed, "client"
    )
    auxiliary["dataset"] = equalize_pairs(
        auxiliary["dataset"], active_pairs, seed, "auxiliary"
    )
    client_matrix = restricted_matrix(client["dataset"], selected_keywords)
    auxiliary_matrix = restricted_matrix(auxiliary["dataset"], selected_keywords)

    previous_cwd = Path.cwd()
    try:
        os.chdir(UPSTREAM)
        real_queries = np.asarray(
            generate_keyword_queries(
                params.gen_params["mode_query"],
                real_frequencies,
                params.gen_params["nqr"],
            )
        )
        observations, _, truth = generate_observations(
            client, params.def_params, real_queries
        )
        predictions = run_attack(
            params.att_params["name"],
            obs=observations,
            aux=auxiliary,
            exp_params=params,
        )
    finally:
        os.chdir(previous_cwd)

    weighted, macro = accuracy(predictions, np.asarray(truth))
    base_client_frequency = frequency_vector(base_client_matrix)
    base_auxiliary_frequency = frequency_vector(base_auxiliary_matrix)
    client_frequency = frequency_vector(client_matrix)
    auxiliary_frequency = frequency_vector(auxiliary_matrix)
    client_pair_equal = all(
        client_frequency[pair["first_position"]]
        == client_frequency[pair["second_position"]]
        for pair in active_pairs
    )
    auxiliary_pair_equal = all(
        auxiliary_frequency[pair["first_position"]]
        == auxiliary_frequency[pair["second_position"]]
        for pair in active_pairs
    )
    return {
        "seed": seed,
        "keyword_count": config["keyword_count"],
        "condition": f"pairs_{pair_count}",
        "equalized_pair_count": pair_count,
        "equalized_keyword_count": 2 * pair_count,
        "equalized_keyword_proportion": 2 * pair_count / config["keyword_count"],
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - tick,
        "selected_keyword_ids": selected_keywords,
        "selected_keyword_ids_sha256": canonical_hash(selected_keywords),
        "base_client_matrix_sha256": sparse_matrix_sha256(base_client_matrix),
        "base_auxiliary_matrix_sha256": sparse_matrix_sha256(base_auxiliary_matrix),
        "intervened_client_matrix_sha256": sparse_matrix_sha256(client_matrix),
        "intervened_auxiliary_matrix_sha256": sparse_matrix_sha256(auxiliary_matrix),
        "base_client_incidence_count": int(base_client_matrix.sum()),
        "base_auxiliary_incidence_count": int(base_auxiliary_matrix.sum()),
        "intervened_client_incidence_count": int(client_matrix.sum()),
        "intervened_auxiliary_incidence_count": int(auxiliary_matrix.sum()),
        "base_client_frequency_sha256": canonical_hash(base_client_frequency.tolist()),
        "base_auxiliary_frequency_sha256": canonical_hash(base_auxiliary_frequency.tolist()),
        "intervened_client_frequency_sha256": canonical_hash(client_frequency.tolist()),
        "intervened_auxiliary_frequency_sha256": canonical_hash(auxiliary_frequency.tolist()),
        "pair_pool": pair_pool,
        "active_pairs": active_pairs,
        "client_active_pairs_frequency_equal": client_pair_equal,
        "auxiliary_active_pairs_frequency_equal": auxiliary_pair_equal,
        "client_active_pair_jaccards": pair_jaccards(client_matrix, [
            {
                "donor_position": pair["first_position"],
                "recipient_position": pair["second_position"],
            }
            for pair in active_pairs
        ]),
        "auxiliary_active_pair_jaccards": pair_jaccards(auxiliary_matrix, [
            {
                "donor_position": pair["first_position"],
                "recipient_position": pair["second_position"],
            }
            for pair in active_pairs
        ]),
        "structure": profile_matrix(client_matrix, structure_config),
        "query_weighted_recovery": weighted,
        "distinct_query_macro_recovery": macro,
        "source": "new_run",
    }


def load_control(config: dict, seed: int) -> dict:
    control = json.loads(
        (ROOT / config["control_results"] / f"seed-{seed}.json").read_text(
            encoding="utf-8"
        )
    )
    (
        _,
        _,
        _,
        _,
        selected_keywords,
        client_matrix,
        auxiliary_matrix,
        pair_pool,
    ) = prepare_base(config, seed)
    client_frequency = frequency_vector(client_matrix)
    auxiliary_frequency = frequency_vector(auxiliary_matrix)
    return {
        "seed": seed,
        "keyword_count": config["keyword_count"],
        "condition": "pairs_0",
        "equalized_pair_count": 0,
        "equalized_keyword_count": 0,
        "equalized_keyword_proportion": 0.0,
        "started_at": control["started_at"],
        "finished_at": control["finished_at"],
        "wall_seconds": control["wall_seconds"],
        "selected_keyword_ids": selected_keywords,
        "selected_keyword_ids_sha256": canonical_hash(selected_keywords),
        "base_client_matrix_sha256": sparse_matrix_sha256(client_matrix),
        "base_auxiliary_matrix_sha256": sparse_matrix_sha256(auxiliary_matrix),
        "intervened_client_matrix_sha256": sparse_matrix_sha256(client_matrix),
        "intervened_auxiliary_matrix_sha256": sparse_matrix_sha256(auxiliary_matrix),
        "base_client_incidence_count": int(client_matrix.sum()),
        "base_auxiliary_incidence_count": int(auxiliary_matrix.sum()),
        "intervened_client_incidence_count": int(client_matrix.sum()),
        "intervened_auxiliary_incidence_count": int(auxiliary_matrix.sum()),
        "base_client_frequency_sha256": canonical_hash(client_frequency.tolist()),
        "base_auxiliary_frequency_sha256": canonical_hash(auxiliary_frequency.tolist()),
        "intervened_client_frequency_sha256": canonical_hash(client_frequency.tolist()),
        "intervened_auxiliary_frequency_sha256": canonical_hash(auxiliary_frequency.tolist()),
        "pair_pool": pair_pool,
        "active_pairs": [],
        "client_active_pairs_frequency_equal": True,
        "auxiliary_active_pairs_frequency_equal": True,
        "client_active_pair_jaccards": [],
        "auxiliary_active_pair_jaccards": [],
        "structure": control["structure"],
        "query_weighted_recovery": control["query_weighted_recovery"],
        "distinct_query_macro_recovery": control["distinct_query_macro_recovery"],
        "source": "reused_control",
    }


def aggregate(records: list[dict], config: dict) -> dict:
    confidence = config["confidence_level"]
    checkpoints = config["attack"]["iteration_checkpoints"]
    pair_counts = config["equalized_pair_counts"]
    record_map = {
        (record["seed"], record["equalized_pair_count"]): record
        for record in records
    }
    by_condition = {}
    for pair_count in pair_counts:
        selected = [
            record
            for record in records
            if record["equalized_pair_count"] == pair_count
        ]
        by_condition[str(pair_count)] = {
            "equalized_keyword_proportion": 2 * pair_count / config["keyword_count"],
            "recovery": {
                str(checkpoint): summary(
                    [record["query_weighted_recovery"][index] for record in selected],
                    confidence,
                )
                for index, checkpoint in enumerate(checkpoints)
            },
            "structure": {
                feature: summary(
                    [record["structure"]["primary_features"][feature] for record in selected],
                    confidence,
                )
                for feature in selected[0]["structure"]["primary_features"]
            },
        }
    paired = {}
    for pair_count in pair_counts[1:]:
        paired[str(pair_count)] = {
            str(checkpoint): summary(
                [
                    record_map[(seed, pair_count)]["query_weighted_recovery"][index]
                    - record_map[(seed, 0)]["query_weighted_recovery"][index]
                    for seed in config["seeds"]
                ],
                confidence,
            )
            for index, checkpoint in enumerate(checkpoints)
        }
    return {"by_condition": by_condition, "paired_differences_from_control": paired}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--pair-count", type=int, action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise RuntimeError(
            f"PYTHONHASHSEED must be {config['python_hash_seed']} before Python starts"
        )
    seeds = config["seeds"] if args.seed is None else args.seed
    pair_counts = (
        config["equalized_pair_counts"]
        if args.pair_count is None
        else args.pair_count
    )
    if not set(seeds).issubset(config["seeds"]):
        raise ValueError("Requested seed is absent from the configuration")
    if not set(pair_counts).issubset(config["equalized_pair_counts"]):
        raise ValueError("Requested pair count is absent from the configuration")

    structure_config = json.loads(
        (ROOT / config["structure_configuration"]).read_text(encoding="utf-8")
    )
    output_directory = ROOT / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    for pair_count in pair_counts:
        directory = output_directory / f"pairs-{pair_count}"
        directory.mkdir(exist_ok=True)
        for seed in seeds:
            output = directory / f"seed-{seed}.json"
            if output.exists():
                print(f"{pair_count} pairs, seed {seed} loaded", flush=True)
            elif pair_count == 0 and config["reuse_control"]:
                record = load_control(config, seed)
                output.write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(f"{pair_count} pairs, seed {seed} reused", flush=True)
            else:
                print(f"{pair_count} pairs, seed {seed} started", flush=True)
                record = run_intervention(config, structure_config, pair_count, seed)
                output.write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(
                    f"{pair_count} pairs, seed {seed} completed in {record['wall_seconds']:.1f} seconds",
                    flush=True,
                )

    expected = [
        output_directory / f"pairs-{pair_count}" / f"seed-{seed}.json"
        for pair_count in config["equalized_pair_counts"]
        for seed in config["seeds"]
    ]
    if not all(path.exists() for path in expected):
        print("Partial run completed", flush=True)
        return
    records = [json.loads(path.read_text(encoding="utf-8")) for path in expected]
    result = {
        "experiment_id": config["id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "dataset_sha256": file_sha256(DATASET),
        "upstream_revision": git_revision(),
        "python": sys.version,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "records": records,
        "aggregate": aggregate(records, config),
    }
    output = output_directory / "aggregate.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
