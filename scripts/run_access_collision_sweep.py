from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "ihop-code"
CONFIG_PATH = ROOT / "configs" / "access_collision_sweep.json"
DATASET = UPSTREAM / "datasets_pro" / "enron-full.pkl"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT / "scripts"))

from defense import generate_observations
from exp_params import ExpParams
from experiment import (
    generate_keyword_queries,
    generate_train_test_data,
    run_attack,
)
from measure_seed_structure import canonical_hash, profile_matrix, restricted_matrix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sparse_matrix_sha256(matrix) -> str:
    canonical = matrix.tocsr(copy=True)
    canonical.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype=np.int64).tobytes())
    digest.update(canonical.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(canonical.indices.astype(np.int64, copy=False).tobytes())
    digest.update(canonical.data.astype(np.int32, copy=False).tobytes())
    return digest.hexdigest()


def git_revision() -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={UPSTREAM.as_posix()}",
        "-C",
        str(UPSTREAM),
        "rev-parse",
        "HEAD",
    ]
    return subprocess.check_output(command, text=True).strip()


def build_params(config: dict) -> ExpParams:
    keyword_count = config["keyword_count"]
    params = ExpParams()
    params.set_general_params(
        dataset="enron-full",
        nkw=keyword_count,
        nqr=keyword_count,
        ndoc=config["selected_documents"],
        freq=config["frequency_source"],
        mode_ds=config["auxiliary_split"],
        mode_fs="past",
        mode_kw="rand",
        mode_query=config["query_mode"],
    )
    params.set_defense_params(config["defence"])
    params.set_attack_params(
        config["attack"]["name"],
        mode=config["attack"]["mode"],
        niters=max(config["attack"]["iteration_checkpoints"]),
        pfree=config["attack"]["free_fraction"],
    )
    params.att_params["niter_list"] = config["attack"][
        "iteration_checkpoints"
    ]
    return params


def collision_pairs(selected_keywords: list[int], pair_count: int) -> list[dict]:
    return [
        {
            "donor_position": 2 * index,
            "recipient_position": 2 * index + 1,
            "donor_keyword_id": int(selected_keywords[2 * index]),
            "recipient_keyword_id": int(selected_keywords[2 * index + 1]),
        }
        for index in range(pair_count)
    ]


def apply_collisions(dataset: list, pairs: list[dict]) -> list[list[int]]:
    transformed = []
    for document in dataset:
        keywords = set(document)
        for pair in pairs:
            donor = pair["donor_keyword_id"]
            recipient = pair["recipient_keyword_id"]
            if donor in keywords:
                keywords.add(recipient)
            else:
                keywords.discard(recipient)
        transformed.append(sorted(keywords))
    return transformed


def accuracy(predictions: list, truth: np.ndarray) -> tuple[list[float], list[float]]:
    weighted = []
    macro = []
    for prediction in predictions:
        correct = np.asarray(
            [query == predicted for query, predicted in zip(truth, prediction)],
            dtype=float,
        )
        weighted.append(float(np.mean(correct)))
        macro.append(
            float(
                np.mean(
                    [np.mean(correct[truth == query]) for query in set(truth)]
                )
            )
        )
    return weighted, macro


def run_intervention(
    config: dict, structure_config: dict, pair_count: int, seed: int
) -> dict:
    params = build_params(config)
    np.random.seed(seed)
    previous_cwd = Path.cwd()
    started = datetime.now(timezone.utc)
    tick = time.perf_counter()
    try:
        os.chdir(UPSTREAM)
        auxiliary, client, real_frequencies = generate_train_test_data(
            params.gen_params
        )
        selected_keywords = [int(value) for value in client["keywords"]]
        pairs = collision_pairs(selected_keywords, pair_count)
        base_client_hash = canonical_hash(client["dataset"])
        base_auxiliary_hash = canonical_hash(auxiliary["dataset"])
        client["dataset"] = apply_collisions(client["dataset"], pairs)
        auxiliary["dataset"] = apply_collisions(auxiliary["dataset"], pairs)

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
    client_matrix = restricted_matrix(client["dataset"], selected_keywords)
    auxiliary_matrix = restricted_matrix(
        auxiliary["dataset"], selected_keywords
    )
    forced_pairs_identical_client = all(
        (
            client_matrix[:, pair["donor_position"]]
            != client_matrix[:, pair["recipient_position"]]
        ).nnz
        == 0
        for pair in pairs
    )
    forced_pairs_identical_auxiliary = all(
        (
            auxiliary_matrix[:, pair["donor_position"]]
            != auxiliary_matrix[:, pair["recipient_position"]]
        ).nnz
        == 0
        for pair in pairs
    )
    return {
        "seed": seed,
        "keyword_count": config["keyword_count"],
        "forced_pair_count": pair_count,
        "forced_keyword_count": 2 * pair_count,
        "forced_keyword_proportion": 2 * pair_count / config["keyword_count"],
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - tick,
        "selected_keyword_ids": selected_keywords,
        "selected_keyword_ids_sha256": canonical_hash(selected_keywords),
        "base_client_documents_sha256": base_client_hash,
        "base_auxiliary_documents_sha256": base_auxiliary_hash,
        "intervened_client_matrix_sha256": sparse_matrix_sha256(client_matrix),
        "intervened_auxiliary_matrix_sha256": sparse_matrix_sha256(
            auxiliary_matrix
        ),
        "collision_pairs": pairs,
        "forced_pairs_identical_client": forced_pairs_identical_client,
        "forced_pairs_identical_auxiliary": forced_pairs_identical_auxiliary,
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
    return {
        "seed": seed,
        "keyword_count": config["keyword_count"],
        "forced_pair_count": 0,
        "forced_keyword_count": 0,
        "forced_keyword_proportion": 0.0,
        "started_at": control["started_at"],
        "finished_at": control["finished_at"],
        "wall_seconds": control["wall_seconds"],
        "selected_keyword_ids": control["selected_keyword_ids"],
        "selected_keyword_ids_sha256": control["selected_keyword_ids_sha256"],
        "base_client_documents_sha256": control["client_documents_sha256"],
        "base_auxiliary_documents_sha256": control[
            "auxiliary_documents_sha256"
        ],
        "intervened_client_matrix_sha256": None,
        "intervened_auxiliary_matrix_sha256": None,
        "collision_pairs": [],
        "forced_pairs_identical_client": True,
        "forced_pairs_identical_auxiliary": True,
        "structure": control["structure"],
        "query_weighted_recovery": control["query_weighted_recovery"],
        "distinct_query_macro_recovery": control[
            "distinct_query_macro_recovery"
        ],
        "source": "reused_control",
    }


def summary(values: list[float], confidence: float) -> dict:
    array = np.asarray(values, dtype=float)
    count = len(array)
    mean = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1))
    critical = float(stats.t.ppf((1 + confidence) / 2, count - 1))
    margin = critical * sample_sd / math.sqrt(count)
    return {
        "n": count,
        "mean": mean,
        "sample_standard_deviation": sample_sd,
        "confidence_level": confidence,
        "confidence_interval_low": mean - margin,
        "confidence_interval_high": mean + margin,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def aggregate(records: list[dict], config: dict) -> dict:
    confidence = config["confidence_level"]
    checkpoints = config["attack"]["iteration_checkpoints"]
    pair_counts = config["forced_pair_counts"]
    record_map = {
        (record["seed"], record["forced_pair_count"]): record
        for record in records
    }
    by_condition = {}
    for pair_count in pair_counts:
        selected = [
            record for record in records if record["forced_pair_count"] == pair_count
        ]
        by_condition[str(pair_count)] = {
            "forced_keyword_proportion": 2
            * pair_count
            / config["keyword_count"],
            "recovery": {
                str(checkpoint): summary(
                    [record["query_weighted_recovery"][index] for record in selected],
                    confidence,
                )
                for index, checkpoint in enumerate(checkpoints)
            },
            "structure": {
                feature: summary(
                    [
                        record["structure"]["primary_features"][feature]
                        for record in selected
                    ],
                    confidence,
                )
                for feature in selected[0]["structure"]["primary_features"]
            },
            "identical_pattern_proportion": summary(
                [
                    record["structure"]["access_pattern_similarity"][
                        "keywords_with_identical_pattern_proportion"
                    ]
                    for record in selected
                ],
                confidence,
            ),
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
        config["forced_pair_counts"]
        if args.pair_count is None
        else args.pair_count
    )
    if not set(seeds).issubset(config["seeds"]):
        raise ValueError("Requested seed is absent from the configuration")
    if not set(pair_counts).issubset(config["forced_pair_counts"]):
        raise ValueError("Requested pair count is absent from the configuration")

    structure_config = json.loads(
        (ROOT / config["structure_configuration"]).read_text(encoding="utf-8")
    )
    output_directory = ROOT / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    for pair_count in pair_counts:
        condition_directory = output_directory / f"pairs-{pair_count}"
        condition_directory.mkdir(exist_ok=True)
        for seed in seeds:
            output = condition_directory / f"seed-{seed}.json"
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
                record = run_intervention(
                    config, structure_config, pair_count, seed
                )
                output.write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(
                    f"{pair_count} pairs, seed {seed} completed in "
                    f"{record['wall_seconds']:.1f} seconds",
                    flush=True,
                )

    expected = [
        output_directory / f"pairs-{pair_count}" / f"seed-{seed}.json"
        for pair_count in config["forced_pair_counts"]
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
        "dataset_sha256": sha256(DATASET),
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
