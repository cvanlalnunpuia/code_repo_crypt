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
CONFIG_PATH = ROOT / "configs" / "sparsity_sweep.json"
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
    sparse_matrix_sha256,
    summary,
)


def local_seed(seed: int, matrix_name: str, keyword_id: int) -> int:
    payload = f"{seed}:{matrix_name}:{keyword_id}:sparsity".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def target_frequency(original: int, retention_rate: float, minimum: int) -> int:
    return min(original, max(minimum, int(round(retention_rate * original))))


def retained_access_sets(
    base_matrix,
    selected_keywords: list[int],
    retention_rate: float,
    minimum: int,
    seed: int,
    matrix_name: str,
) -> list[list[int]]:
    retained = []
    for position, keyword_id in enumerate(selected_keywords):
        original_documents = base_matrix[:, position].nonzero()[0]
        generator = np.random.RandomState(
            local_seed(seed, matrix_name, keyword_id)
        )
        ordered_documents = generator.permutation(original_documents)
        count = target_frequency(
            len(original_documents), retention_rate, minimum
        )
        retained.append(sorted(int(value) for value in ordered_documents[:count]))
    return retained


def apply_retention(
    dataset: list,
    selected_keywords: list[int],
    retained: list[list[int]],
) -> list[list[int]]:
    documents = [set(document) for document in dataset]
    for keyword_id, retained_documents in zip(selected_keywords, retained):
        for document in documents:
            document.discard(keyword_id)
        for document_index in retained_documents:
            documents[document_index].add(keyword_id)
    return [sorted(document) for document in documents]


def weak_frequency_order_preserved(
    original: np.ndarray, intervened: np.ndarray
) -> bool:
    order = np.argsort(original, kind="stable")
    ordered_original = original[order]
    ordered_intervened = intervened[order]
    for index in range(len(order) - 1):
        if (
            ordered_original[index] < ordered_original[index + 1]
            and ordered_intervened[index] > ordered_intervened[index + 1]
        ):
            return False
    return True


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
    return (
        params,
        auxiliary,
        client,
        real_frequencies,
        selected_keywords,
        client_matrix,
        auxiliary_matrix,
    )


def run_intervention(
    config: dict, structure_config: dict, retention_rate: float, seed: int
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
    ) = prepare_base(config, seed)
    minimum = config["minimum_column_frequency"]
    client_retained = retained_access_sets(
        base_client_matrix,
        selected_keywords,
        retention_rate,
        minimum,
        seed,
        "client",
    )
    auxiliary_retained = retained_access_sets(
        base_auxiliary_matrix,
        selected_keywords,
        retention_rate,
        minimum,
        seed,
        "auxiliary",
    )
    client["dataset"] = apply_retention(
        client["dataset"], selected_keywords, client_retained
    )
    auxiliary["dataset"] = apply_retention(
        auxiliary["dataset"], selected_keywords, auxiliary_retained
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
    return {
        "seed": seed,
        "keyword_count": config["keyword_count"],
        "condition": f"retention_{retention_rate:.2f}",
        "retention_rate": retention_rate,
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
        "achieved_client_incidence_retention": float(
            client_matrix.sum() / base_client_matrix.sum()
        ),
        "achieved_auxiliary_incidence_retention": float(
            auxiliary_matrix.sum() / base_auxiliary_matrix.sum()
        ),
        "base_client_frequency_sha256": canonical_hash(base_client_frequency.tolist()),
        "base_auxiliary_frequency_sha256": canonical_hash(base_auxiliary_frequency.tolist()),
        "intervened_client_frequency_sha256": canonical_hash(client_frequency.tolist()),
        "intervened_auxiliary_frequency_sha256": canonical_hash(auxiliary_frequency.tolist()),
        "client_retained_access_sha256": canonical_hash(client_retained),
        "auxiliary_retained_access_sha256": canonical_hash(auxiliary_retained),
        "client_minimum_frequency": int(np.min(client_frequency)),
        "auxiliary_minimum_frequency": int(np.min(auxiliary_frequency)),
        "client_frequency_order_preserved": weak_frequency_order_preserved(
            base_client_frequency, client_frequency
        ),
        "auxiliary_frequency_order_preserved": weak_frequency_order_preserved(
            base_auxiliary_frequency, auxiliary_frequency
        ),
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
    ) = prepare_base(config, seed)
    client_frequency = frequency_vector(client_matrix)
    auxiliary_frequency = frequency_vector(auxiliary_matrix)
    client_retained = [
        sorted(int(value) for value in client_matrix[:, position].nonzero()[0])
        for position in range(config["keyword_count"])
    ]
    auxiliary_retained = [
        sorted(int(value) for value in auxiliary_matrix[:, position].nonzero()[0])
        for position in range(config["keyword_count"])
    ]
    return {
        "seed": seed,
        "keyword_count": config["keyword_count"],
        "condition": "retention_1.00",
        "retention_rate": 1.0,
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
        "achieved_client_incidence_retention": 1.0,
        "achieved_auxiliary_incidence_retention": 1.0,
        "base_client_frequency_sha256": canonical_hash(client_frequency.tolist()),
        "base_auxiliary_frequency_sha256": canonical_hash(auxiliary_frequency.tolist()),
        "intervened_client_frequency_sha256": canonical_hash(client_frequency.tolist()),
        "intervened_auxiliary_frequency_sha256": canonical_hash(auxiliary_frequency.tolist()),
        "client_retained_access_sha256": canonical_hash(client_retained),
        "auxiliary_retained_access_sha256": canonical_hash(auxiliary_retained),
        "client_minimum_frequency": int(np.min(client_frequency)),
        "auxiliary_minimum_frequency": int(np.min(auxiliary_frequency)),
        "client_frequency_order_preserved": True,
        "auxiliary_frequency_order_preserved": True,
        "structure": control["structure"],
        "query_weighted_recovery": control["query_weighted_recovery"],
        "distinct_query_macro_recovery": control["distinct_query_macro_recovery"],
        "source": "reused_control",
    }


def aggregate(records: list[dict], config: dict) -> dict:
    confidence = config["confidence_level"]
    checkpoints = config["attack"]["iteration_checkpoints"]
    rates = config["retention_rates"]
    record_map = {
        (record["seed"], record["retention_rate"]): record for record in records
    }
    by_condition = {}
    for rate in rates:
        selected = [record for record in records if record["retention_rate"] == rate]
        by_condition[f"{rate:.2f}"] = {
            "retention_rate": rate,
            "achieved_client_incidence_retention": summary(
                [record["achieved_client_incidence_retention"] for record in selected],
                confidence,
            ),
            "achieved_auxiliary_incidence_retention": summary(
                [record["achieved_auxiliary_incidence_retention"] for record in selected],
                confidence,
            ),
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
    for rate in rates[1:]:
        paired[f"{rate:.2f}"] = {
            str(checkpoint): summary(
                [
                    record_map[(seed, rate)]["query_weighted_recovery"][index]
                    - record_map[(seed, 1.0)]["query_weighted_recovery"][index]
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
    parser.add_argument("--retention", type=float, action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise RuntimeError(
            f"PYTHONHASHSEED must be {config['python_hash_seed']} before Python starts"
        )
    seeds = config["seeds"] if args.seed is None else args.seed
    rates = config["retention_rates"] if args.retention is None else args.retention
    if not set(seeds).issubset(config["seeds"]):
        raise ValueError("Requested seed is absent from the configuration")
    if not set(rates).issubset(config["retention_rates"]):
        raise ValueError("Requested retention is absent from the configuration")

    structure_config = json.loads(
        (ROOT / config["structure_configuration"]).read_text(encoding="utf-8")
    )
    output_directory = ROOT / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    for rate in rates:
        directory = output_directory / f"retention-{rate:.2f}"
        directory.mkdir(exist_ok=True)
        for seed in seeds:
            output = directory / f"seed-{seed}.json"
            if output.exists():
                print(f"retention {rate:.2f}, seed {seed} loaded", flush=True)
            elif rate == 1.0 and config["reuse_full_retention_control"]:
                record = load_control(config, seed)
                output.write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(f"retention {rate:.2f}, seed {seed} reused", flush=True)
            else:
                print(f"retention {rate:.2f}, seed {seed} started", flush=True)
                record = run_intervention(config, structure_config, rate, seed)
                output.write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(
                    f"retention {rate:.2f}, seed {seed} completed in {record['wall_seconds']:.1f} seconds",
                    flush=True,
                )

    expected = [
        output_directory / f"retention-{rate:.2f}" / f"seed-{seed}.json"
        for rate in config["retention_rates"]
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
