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
from typing import Optional

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "ihop-code"
CONFIG_PATH = ROOT / "configs" / "access_similarity_sweep.json"
DATASET = UPSTREAM / "datasets_pro" / "enron-full.pkl"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT / "scripts"))

from defense import generate_observations
from exp_params import ExpParams
from experiment import generate_keyword_queries, generate_train_test_data, run_attack
from measure_seed_structure import canonical_hash, profile_matrix, restricted_matrix


def file_sha256(path: Path) -> str:
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
    params.att_params["niter_list"] = config["attack"]["iteration_checkpoints"]
    return params


def frequency_vector(matrix) -> np.ndarray:
    return np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)


def select_pairs(
    selected_keywords: list[int],
    client_frequency: np.ndarray,
    auxiliary_frequency: np.ndarray,
    pair_count: int,
    maximum_target: float,
) -> list[dict]:
    candidates = []
    keyword_count = len(selected_keywords)
    for first in range(keyword_count):
        for second in range(first + 1, keyword_count):
            client_first = int(client_frequency[first])
            client_second = int(client_frequency[second])
            auxiliary_first = int(auxiliary_frequency[first])
            auxiliary_second = int(auxiliary_frequency[second])
            if min(client_first, client_second, auxiliary_first, auxiliary_second) == 0:
                continue
            if client_first == client_second or auxiliary_first == auxiliary_second:
                continue
            client_ratio = min(client_first, client_second) / max(
                client_first, client_second
            )
            auxiliary_ratio = min(auxiliary_first, auxiliary_second) / max(
                auxiliary_first, auxiliary_second
            )
            if min(client_ratio, auxiliary_ratio) < maximum_target:
                continue
            score = abs(math.log(client_first / client_second)) + abs(
                math.log(auxiliary_first / auxiliary_second)
            )
            candidates.append((score, first, second))

    candidates.sort()
    selected = []
    used = set()
    for score, donor, recipient in candidates:
        if donor in used or recipient in used:
            continue
        used.update((donor, recipient))
        selected.append(
            {
                "donor_position": donor,
                "recipient_position": recipient,
                "donor_keyword_id": int(selected_keywords[donor]),
                "recipient_keyword_id": int(selected_keywords[recipient]),
                "selection_score": score,
                "client_donor_frequency": int(client_frequency[donor]),
                "client_recipient_frequency": int(client_frequency[recipient]),
                "auxiliary_donor_frequency": int(auxiliary_frequency[donor]),
                "auxiliary_recipient_frequency": int(auxiliary_frequency[recipient]),
            }
        )
        if len(selected) == pair_count:
            return selected
    raise RuntimeError(
        f"Only {len(selected)} disjoint pairs support target Jaccard {maximum_target}"
    )


def local_seed(seed: int, matrix_name: str, donor: int, recipient: int) -> int:
    payload = f"{seed}:{matrix_name}:{donor}:{recipient}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def target_recipient_documents(
    document_count: int,
    donor_documents: np.ndarray,
    recipient_count: int,
    target_jaccard: float,
    seed: int,
) -> np.ndarray:
    donor_count = len(donor_documents)
    desired_overlap = int(
        round(target_jaccard * (donor_count + recipient_count) / (1 + target_jaccard))
    )
    overlap = min(desired_overlap, donor_count, recipient_count)
    outside_count = recipient_count - overlap
    donor_mask = np.zeros(document_count, dtype=bool)
    donor_mask[donor_documents] = True
    outside_documents = np.flatnonzero(~donor_mask)
    if outside_count > len(outside_documents):
        raise RuntimeError("Insufficient documents outside the donor access pattern")
    generator = np.random.RandomState(seed)
    selected_overlap = generator.permutation(donor_documents)[:overlap]
    selected_outside = generator.permutation(outside_documents)[:outside_count]
    return np.sort(np.concatenate((selected_overlap, selected_outside)))


def apply_similarity(
    dataset: list,
    base_matrix,
    pairs: list[dict],
    target_jaccard: float,
    seed: int,
    matrix_name: str,
) -> list[list[int]]:
    documents = [set(document) for document in dataset]
    document_count = len(documents)
    for pair in pairs:
        donor_position = pair["donor_position"]
        recipient_position = pair["recipient_position"]
        recipient_keyword = pair["recipient_keyword_id"]
        donor_documents = base_matrix[:, donor_position].nonzero()[0]
        recipient_count = int(base_matrix[:, recipient_position].sum())
        selected_documents = target_recipient_documents(
            document_count,
            donor_documents,
            recipient_count,
            target_jaccard,
            local_seed(
                seed,
                matrix_name,
                pair["donor_keyword_id"],
                recipient_keyword,
            ),
        )
        for document in documents:
            document.discard(recipient_keyword)
        for document_index in selected_documents:
            documents[int(document_index)].add(recipient_keyword)
    return [sorted(document) for document in documents]


def pair_jaccards(matrix, pairs: list[dict]) -> list[float]:
    values = []
    for pair in pairs:
        donor = matrix[:, pair["donor_position"]]
        recipient = matrix[:, pair["recipient_position"]]
        intersection = donor.multiply(recipient).sum()
        union = donor.sum() + recipient.sum() - intersection
        values.append(float(intersection / union))
    return values


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
            float(np.mean([np.mean(correct[truth == query]) for query in set(truth)]))
        )
    return weighted, macro


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
        max(config["target_jaccard_values"]),
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
    config: dict, structure_config: dict, target_jaccard: float, seed: int
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

    client["dataset"] = apply_similarity(
        client["dataset"],
        base_client_matrix,
        pairs,
        target_jaccard,
        seed,
        "client",
    )
    auxiliary["dataset"] = apply_similarity(
        auxiliary["dataset"],
        base_auxiliary_matrix,
        pairs,
        target_jaccard,
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
    client_jaccards = pair_jaccards(client_matrix, pairs)
    auxiliary_jaccards = pair_jaccards(auxiliary_matrix, pairs)
    return {
        "seed": seed,
        "keyword_count": config["keyword_count"],
        "pair_count": config["pair_count"],
        "condition": f"target_{target_jaccard:.2f}",
        "target_jaccard": target_jaccard,
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
        "client_frequency_preserved": np.array_equal(
            base_client_frequency, client_frequency
        ),
        "auxiliary_frequency_preserved": np.array_equal(
            base_auxiliary_frequency, auxiliary_frequency
        ),
        "pairs": pairs,
        "client_pair_jaccards": client_jaccards,
        "auxiliary_pair_jaccards": auxiliary_jaccards,
        "client_pair_jaccard_mean": float(np.mean(client_jaccards)),
        "auxiliary_pair_jaccard_mean": float(np.mean(auxiliary_jaccards)),
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
        pairs,
    ) = prepare_base(config, seed)
    client_frequency = frequency_vector(client_matrix)
    auxiliary_frequency = frequency_vector(auxiliary_matrix)
    return {
        "seed": seed,
        "keyword_count": config["keyword_count"],
        "pair_count": config["pair_count"],
        "condition": "control",
        "target_jaccard": None,
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
        "pairs": pairs,
        "client_pair_jaccards": pair_jaccards(client_matrix, pairs),
        "auxiliary_pair_jaccards": pair_jaccards(auxiliary_matrix, pairs),
        "client_pair_jaccard_mean": float(np.mean(pair_jaccards(client_matrix, pairs))),
        "auxiliary_pair_jaccard_mean": float(
            np.mean(pair_jaccards(auxiliary_matrix, pairs))
        ),
        "structure": control["structure"],
        "query_weighted_recovery": control["query_weighted_recovery"],
        "distinct_query_macro_recovery": control["distinct_query_macro_recovery"],
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
    conditions = ["control"] + [
        f"target_{target:.2f}" for target in config["target_jaccard_values"]
    ]
    record_map = {(record["seed"], record["condition"]): record for record in records}
    by_condition = {}
    for condition in conditions:
        selected = [record for record in records if record["condition"] == condition]
        by_condition[condition] = {
            "target_jaccard": selected[0]["target_jaccard"],
            "client_pair_jaccard": summary(
                [record["client_pair_jaccard_mean"] for record in selected], confidence
            ),
            "auxiliary_pair_jaccard": summary(
                [record["auxiliary_pair_jaccard_mean"] for record in selected],
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
    for condition in conditions[1:]:
        paired[condition] = {
            str(checkpoint): summary(
                [
                    record_map[(seed, condition)]["query_weighted_recovery"][index]
                    - record_map[(seed, "control")]["query_weighted_recovery"][index]
                    for seed in config["seeds"]
                ],
                confidence,
            )
            for index, checkpoint in enumerate(checkpoints)
        }
    return {"by_condition": by_condition, "paired_differences_from_control": paired}


def condition_directory(target: Optional[float]) -> str:
    return "control" if target is None else f"target-{target:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--target", type=float, action="append")
    parser.add_argument("--include-control", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise RuntimeError(
            f"PYTHONHASHSEED must be {config['python_hash_seed']} before Python starts"
        )
    seeds = config["seeds"] if args.seed is None else args.seed
    targets = config["target_jaccard_values"] if args.target is None else args.target
    if not set(seeds).issubset(config["seeds"]):
        raise ValueError("Requested seed is absent from the configuration")
    if not set(targets).issubset(config["target_jaccard_values"]):
        raise ValueError("Requested target is absent from the configuration")
    run_control = args.include_control or args.seed is None

    structure_config = json.loads(
        (ROOT / config["structure_configuration"]).read_text(encoding="utf-8")
    )
    output_directory = ROOT / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    conditions = ([None] if run_control else []) + targets
    for target in conditions:
        directory = output_directory / condition_directory(target)
        directory.mkdir(exist_ok=True)
        for seed in seeds:
            output = directory / f"seed-{seed}.json"
            label = "control" if target is None else f"target {target:.2f}"
            if output.exists():
                print(f"{label}, seed {seed} loaded", flush=True)
            elif target is None and config["reuse_control"]:
                record = load_control(config, seed)
                output.write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(f"{label}, seed {seed} reused", flush=True)
            else:
                print(f"{label}, seed {seed} started", flush=True)
                record = run_intervention(config, structure_config, target, seed)
                output.write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(
                    f"{label}, seed {seed} completed in {record['wall_seconds']:.1f} seconds",
                    flush=True,
                )

    all_targets = [None] + config["target_jaccard_values"]
    expected = [
        output_directory / condition_directory(target) / f"seed-{seed}.json"
        for target in all_targets
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
