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
from typing import Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "ihop-code"
CONFIG_PATH = ROOT / "configs" / "cooccurrence_similarity_sweep.json"
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
    select_pairs,
    sparse_matrix_sha256,
    summary,
)


def local_seed(seed: int, matrix_name: str, donor: int, recipient: int) -> int:
    payload = f"{seed}:{matrix_name}:{donor}:{recipient}:cooccurrence".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def sample_candidate_set(
    generator: np.random.RandomState,
    candidates: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    selected_indices = []
    selected_set = set()
    while len(selected_indices) < sample_size:
        remaining = sample_size - len(selected_indices)
        draws = generator.randint(0, len(candidates), size=max(remaining * 2, 16))
        for draw in draws:
            index = int(draw)
            if index not in selected_set:
                selected_set.add(index)
                selected_indices.append(index)
                if len(selected_indices) == sample_size:
                    break
    return candidates[np.asarray(selected_indices, dtype=int)]


def apply_profile_matching(
    dataset: list,
    base_matrix,
    pairs: list[dict],
    candidate_budget: int,
    seed: int,
    matrix_name: str,
) -> tuple[list[list[int]], list[dict]]:
    documents = [set(document) for document in dataset]
    document_count = len(documents)
    metadata = []
    for pair in pairs:
        donor_position = pair["donor_position"]
        recipient_position = pair["recipient_position"]
        donor_keyword = pair["donor_keyword_id"]
        recipient_keyword = pair["recipient_keyword_id"]
        recipient_count = int(base_matrix[:, recipient_position].sum())
        donor_documents = base_matrix[:, donor_position].nonzero()[0]
        donor_mask = np.zeros(document_count, dtype=bool)
        donor_mask[donor_documents] = True
        candidates = np.flatnonzero(~donor_mask)
        donor_profile = np.asarray(
            base_matrix[donor_documents, :].sum(axis=0)
        ).ravel().astype(float)
        donor_profile[donor_position] = 0.0
        donor_profile[recipient_position] = 0.0
        donor_norm = np.linalg.norm(donor_profile)
        generator = np.random.RandomState(
            local_seed(seed, matrix_name, donor_keyword, recipient_keyword)
        )
        best_documents = None
        best_cosine = -1.0
        for _ in range(candidate_budget):
            proposed_documents = sample_candidate_set(
                generator, candidates, recipient_count
            )
            proposed_profile = np.asarray(
                base_matrix[proposed_documents, :].sum(axis=0)
            ).ravel().astype(float)
            proposed_profile[donor_position] = 0.0
            proposed_profile[recipient_position] = 0.0
            denominator = donor_norm * np.linalg.norm(proposed_profile)
            cosine = float(np.dot(donor_profile, proposed_profile) / denominator)
            if cosine > best_cosine:
                best_cosine = cosine
                best_documents = proposed_documents
        if best_documents is None:
            raise RuntimeError("No candidate access set was evaluated")

        for document in documents:
            document.discard(recipient_keyword)
        for document_index in best_documents:
            documents[int(document_index)].add(recipient_keyword)
        metadata.append(
            {
                "donor_position": donor_position,
                "recipient_position": recipient_position,
                "recipient_frequency": recipient_count,
                "candidate_set_budget": candidate_budget,
                "selected_base_profile_cosine": best_cosine,
            }
        )
    return [sorted(document) for document in documents], metadata


def pair_cooccurrence_cosines(matrix, pairs: list[dict]) -> list[float]:
    cooccurrence = (matrix.T @ matrix).astype(float).toarray()
    np.fill_diagonal(cooccurrence, 0.0)
    values = []
    for pair in pairs:
        first = cooccurrence[pair["donor_position"]]
        second = cooccurrence[pair["recipient_position"]]
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        values.append(float(np.dot(first, second) / denominator))
    return values


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
    pairs = select_pairs(
        selected_keywords,
        frequency_vector(client_matrix),
        frequency_vector(auxiliary_matrix),
        config["pair_count"],
        0.75,
    )
    return (
        params,
        auxiliary,
        client,
        real_frequencies,
        selected_keywords,
        client_matrix,
        auxiliary_matrix,
        pairs,
    )


def run_intervention(
    config: dict, structure_config: dict, candidate_budget: int, seed: int
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
        pairs,
    ) = prepare_base(config, seed)
    client["dataset"], client_matching = apply_profile_matching(
        client["dataset"],
        base_client_matrix,
        pairs,
        candidate_budget,
        seed,
        "client",
    )
    auxiliary["dataset"], auxiliary_matching = apply_profile_matching(
        auxiliary["dataset"],
        base_auxiliary_matrix,
        pairs,
        candidate_budget,
        seed,
        "auxiliary",
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
        "condition": f"budget_{candidate_budget}",
        "candidate_set_budget": candidate_budget,
        "pair_count": config["pair_count"],
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - tick,
        "selected_keyword_ids": selected_keywords,
        "selected_keyword_ids_sha256": canonical_hash(selected_keywords),
        "base_client_matrix_sha256": sparse_matrix_sha256(base_client_matrix),
        "base_auxiliary_matrix_sha256": sparse_matrix_sha256(base_auxiliary_matrix),
        "intervened_client_matrix_sha256": sparse_matrix_sha256(client_matrix),
        "intervened_auxiliary_matrix_sha256": sparse_matrix_sha256(auxiliary_matrix),
        "base_client_frequency_sha256": canonical_hash(base_client_frequency.tolist()),
        "base_auxiliary_frequency_sha256": canonical_hash(base_auxiliary_frequency.tolist()),
        "intervened_client_frequency_sha256": canonical_hash(client_frequency.tolist()),
        "intervened_auxiliary_frequency_sha256": canonical_hash(auxiliary_frequency.tolist()),
        "client_frequency_preserved": np.array_equal(base_client_frequency, client_frequency),
        "auxiliary_frequency_preserved": np.array_equal(base_auxiliary_frequency, auxiliary_frequency),
        "base_client_incidence_count": int(base_client_matrix.sum()),
        "base_auxiliary_incidence_count": int(base_auxiliary_matrix.sum()),
        "intervened_client_incidence_count": int(client_matrix.sum()),
        "intervened_auxiliary_incidence_count": int(auxiliary_matrix.sum()),
        "pairs": pairs,
        "client_matching": client_matching,
        "auxiliary_matching": auxiliary_matching,
        "client_pair_access_jaccards": pair_jaccards(client_matrix, pairs),
        "auxiliary_pair_access_jaccards": pair_jaccards(auxiliary_matrix, pairs),
        "client_pair_cooccurrence_cosines": pair_cooccurrence_cosines(client_matrix, pairs),
        "auxiliary_pair_cooccurrence_cosines": pair_cooccurrence_cosines(auxiliary_matrix, pairs),
        "client_pair_cooccurrence_cosine_mean": float(
            np.mean(pair_cooccurrence_cosines(client_matrix, pairs))
        ),
        "auxiliary_pair_cooccurrence_cosine_mean": float(
            np.mean(pair_cooccurrence_cosines(auxiliary_matrix, pairs))
        ),
        "structure": profile_matrix(client_matrix, structure_config),
        "query_weighted_recovery": weighted,
        "distinct_query_macro_recovery": macro,
        "source": "new_run",
    }


def load_original_control(config: dict, seed: int) -> dict:
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
        pairs,
    ) = prepare_base(config, seed)
    client_frequency = frequency_vector(client_matrix)
    auxiliary_frequency = frequency_vector(auxiliary_matrix)
    return {
        "seed": seed,
        "keyword_count": config["keyword_count"],
        "condition": "original",
        "candidate_set_budget": None,
        "pair_count": config["pair_count"],
        "started_at": control["started_at"],
        "finished_at": control["finished_at"],
        "wall_seconds": control["wall_seconds"],
        "selected_keyword_ids": selected_keywords,
        "selected_keyword_ids_sha256": canonical_hash(selected_keywords),
        "base_client_matrix_sha256": sparse_matrix_sha256(client_matrix),
        "base_auxiliary_matrix_sha256": sparse_matrix_sha256(auxiliary_matrix),
        "intervened_client_matrix_sha256": sparse_matrix_sha256(client_matrix),
        "intervened_auxiliary_matrix_sha256": sparse_matrix_sha256(auxiliary_matrix),
        "base_client_frequency_sha256": canonical_hash(client_frequency.tolist()),
        "base_auxiliary_frequency_sha256": canonical_hash(auxiliary_frequency.tolist()),
        "intervened_client_frequency_sha256": canonical_hash(client_frequency.tolist()),
        "intervened_auxiliary_frequency_sha256": canonical_hash(auxiliary_frequency.tolist()),
        "client_frequency_preserved": True,
        "auxiliary_frequency_preserved": True,
        "base_client_incidence_count": int(client_matrix.sum()),
        "base_auxiliary_incidence_count": int(auxiliary_matrix.sum()),
        "intervened_client_incidence_count": int(client_matrix.sum()),
        "intervened_auxiliary_incidence_count": int(auxiliary_matrix.sum()),
        "pairs": pairs,
        "client_matching": [],
        "auxiliary_matching": [],
        "client_pair_access_jaccards": pair_jaccards(client_matrix, pairs),
        "auxiliary_pair_access_jaccards": pair_jaccards(auxiliary_matrix, pairs),
        "client_pair_cooccurrence_cosines": pair_cooccurrence_cosines(client_matrix, pairs),
        "auxiliary_pair_cooccurrence_cosines": pair_cooccurrence_cosines(auxiliary_matrix, pairs),
        "client_pair_cooccurrence_cosine_mean": float(
            np.mean(pair_cooccurrence_cosines(client_matrix, pairs))
        ),
        "auxiliary_pair_cooccurrence_cosine_mean": float(
            np.mean(pair_cooccurrence_cosines(auxiliary_matrix, pairs))
        ),
        "structure": control["structure"],
        "query_weighted_recovery": control["query_weighted_recovery"],
        "distinct_query_macro_recovery": control["distinct_query_macro_recovery"],
        "source": "reused_original_control",
    }


def aggregate(records: list[dict], config: dict) -> dict:
    confidence = config["confidence_level"]
    checkpoints = config["attack"]["iteration_checkpoints"]
    conditions = ["original"] + [
        f"budget_{budget}" for budget in config["candidate_set_budgets"]
    ]
    record_map = {(record["seed"], record["condition"]): record for record in records}
    by_condition = {}
    for condition in conditions:
        selected = [record for record in records if record["condition"] == condition]
        by_condition[condition] = {
            "candidate_set_budget": selected[0]["candidate_set_budget"],
            "client_pair_cooccurrence_cosine": summary(
                [record["client_pair_cooccurrence_cosine_mean"] for record in selected],
                confidence,
            ),
            "auxiliary_pair_cooccurrence_cosine": summary(
                [record["auxiliary_pair_cooccurrence_cosine_mean"] for record in selected],
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

    paired_from_random = {}
    random_condition = "budget_1"
    for condition in conditions[2:]:
        paired_from_random[condition] = {
            str(checkpoint): summary(
                [
                    record_map[(seed, condition)]["query_weighted_recovery"][index]
                    - record_map[(seed, random_condition)]["query_weighted_recovery"][index]
                    for seed in config["seeds"]
                ],
                confidence,
            )
            for index, checkpoint in enumerate(checkpoints)
        }
    return {
        "by_condition": by_condition,
        "paired_differences_from_random_reconstruction": paired_from_random,
    }


def directory_name(budget: Optional[int]) -> str:
    return "original" if budget is None else f"budget-{budget}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--budget", type=int, action="append")
    parser.add_argument("--include-original", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise RuntimeError(
            f"PYTHONHASHSEED must be {config['python_hash_seed']} before Python starts"
        )
    seeds = config["seeds"] if args.seed is None else args.seed
    budgets = (
        config["candidate_set_budgets"]
        if args.budget is None
        else args.budget
    )
    if not set(seeds).issubset(config["seeds"]):
        raise ValueError("Requested seed is absent from the configuration")
    if not set(budgets).issubset(config["candidate_set_budgets"]):
        raise ValueError("Requested budget is absent from the configuration")
    run_original = args.include_original or args.seed is None

    structure_config = json.loads(
        (ROOT / config["structure_configuration"]).read_text(encoding="utf-8")
    )
    output_directory = ROOT / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    conditions = ([None] if run_original else []) + budgets
    for budget in conditions:
        directory = output_directory / directory_name(budget)
        directory.mkdir(exist_ok=True)
        for seed in seeds:
            output = directory / f"seed-{seed}.json"
            label = "original" if budget is None else f"budget {budget}"
            if output.exists():
                print(f"{label}, seed {seed} loaded", flush=True)
            elif budget is None and config["reuse_original_control"]:
                record = load_original_control(config, seed)
                output.write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(f"{label}, seed {seed} reused", flush=True)
            else:
                print(f"{label}, seed {seed} started", flush=True)
                record = run_intervention(config, structure_config, budget, seed)
                output.write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(
                    f"{label}, seed {seed} completed in {record['wall_seconds']:.1f} seconds",
                    flush=True,
                )

    all_budgets = [None] + config["candidate_set_budgets"]
    expected = [
        output_directory / directory_name(budget) / f"seed-{seed}.json"
        for budget in all_budgets
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
