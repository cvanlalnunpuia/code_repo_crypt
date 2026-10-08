from __future__ import annotations

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
CONFIG_PATH = ROOT / "configs" / "keyword_size_sweep.json"
DATASET = UPSTREAM / "datasets_pro" / "enron-full.pkl"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT / "scripts"))

from exp_params import ExpParams
from experiment import generate_train_test_data, run_experiment
from measure_seed_structure import canonical_hash, profile_matrix, restricted_matrix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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


def build_params(config: dict, keyword_count: int) -> ExpParams:
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


def reproduce_selection(params: ExpParams, seed: int):
    np.random.seed(seed)
    previous_cwd = Path.cwd()
    try:
        os.chdir(UPSTREAM)
        return generate_train_test_data(params.gen_params)
    finally:
        os.chdir(previous_cwd)


def run_new(config: dict, structure_config: dict, keyword_count: int, seed: int) -> dict:
    params = build_params(config, keyword_count)
    started = datetime.now(timezone.utc)
    tick = time.perf_counter()
    previous_cwd = Path.cwd()
    try:
        os.chdir(UPSTREAM)
        query_weighted, distinct_macro, upstream_seconds = run_experiment(
            params, seed=seed, debug_mode=False
        )
    finally:
        os.chdir(previous_cwd)

    auxiliary, client, _ = reproduce_selection(params, seed)
    selected_keywords = [int(value) for value in client["keywords"]]
    matrix = restricted_matrix(client["dataset"], selected_keywords)
    return {
        "seed": seed,
        "keyword_count": keyword_count,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - tick,
        "upstream_seconds": float(upstream_seconds),
        "selected_keyword_ids": selected_keywords,
        "selected_keyword_ids_sha256": canonical_hash(selected_keywords),
        "client_documents_sha256": canonical_hash(client["dataset"]),
        "auxiliary_documents_sha256": canonical_hash(auxiliary["dataset"]),
        "structure": profile_matrix(matrix, structure_config),
        "query_weighted_recovery": [float(value) for value in query_weighted],
        "distinct_query_macro_recovery": [
            float(value) for value in distinct_macro
        ],
        "source": "new_run",
    }


def load_500(config: dict, seed: int) -> dict:
    baseline = json.loads(
        (ROOT / config["baseline_results"] / f"seed-{seed}.json").read_text(
            encoding="utf-8"
        )
    )
    structure = json.loads(
        (ROOT / config["baseline_structure"] / f"seed-{seed}.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "seed": seed,
        "keyword_count": 500,
        "started_at": baseline["started_at"],
        "finished_at": baseline["finished_at"],
        "wall_seconds": baseline["wall_seconds"],
        "upstream_seconds": baseline["upstream_seconds"],
        "selected_keyword_ids": structure["selected_keyword_ids"],
        "selected_keyword_ids_sha256": structure[
            "selected_keyword_ids_sha256"
        ],
        "client_documents_sha256": structure["client_documents_sha256"],
        "auxiliary_documents_sha256": structure["auxiliary_documents_sha256"],
        "structure": structure["structure"],
        "query_weighted_recovery": baseline["query_weighted_recovery"],
        "distinct_query_macro_recovery": baseline[
            "distinct_query_macro_recovery"
        ],
        "source": "reused_baseline",
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
    by_size = {}
    for keyword_count in config["keyword_counts"]:
        selected = [r for r in records if r["keyword_count"] == keyword_count]
        by_size[str(keyword_count)] = {
            "recovery": {
                str(checkpoint): summary(
                    [r["query_weighted_recovery"][index] for r in selected],
                    confidence,
                )
                for index, checkpoint in enumerate(checkpoints)
            },
            "structure": {
                feature: summary(
                    [r["structure"]["primary_features"][feature] for r in selected],
                    confidence,
                )
                for feature in selected[0]["structure"]["primary_features"]
            },
        }

    paired = {}
    sizes = config["keyword_counts"]
    record_map = {(r["seed"], r["keyword_count"]): r for r in records}
    for smaller, larger in zip(sizes, sizes[1:]):
        pair_name = f"{larger}_minus_{smaller}"
        paired[pair_name] = {
            str(checkpoint): summary(
                [
                    record_map[(seed, larger)]["query_weighted_recovery"][index]
                    - record_map[(seed, smaller)]["query_weighted_recovery"][index]
                    for seed in config["seeds"]
                ],
                confidence,
            )
            for index, checkpoint in enumerate(checkpoints)
        }
    paired[f"{sizes[-1]}_minus_{sizes[0]}"] = {
        str(checkpoint): summary(
            [
                record_map[(seed, sizes[-1])]["query_weighted_recovery"][index]
                - record_map[(seed, sizes[0])]["query_weighted_recovery"][index]
                for seed in config["seeds"]
            ],
            confidence,
        )
        for index, checkpoint in enumerate(checkpoints)
    }
    return {"by_keyword_count": by_size, "paired_recovery_differences": paired}


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    required_hash_seed = str(config["python_hash_seed"])
    if os.environ.get("PYTHONHASHSEED") != required_hash_seed:
        raise RuntimeError(
            f"PYTHONHASHSEED must be {required_hash_seed} before Python starts"
        )
    structure_config = json.loads(
        (ROOT / config["structure_configuration"]).read_text(encoding="utf-8")
    )
    output_directory = ROOT / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    records = []

    for keyword_count in config["keyword_counts"]:
        size_directory = output_directory / f"keywords-{keyword_count}"
        size_directory.mkdir(exist_ok=True)
        for seed in config["seeds"]:
            output = size_directory / f"seed-{seed}.json"
            if output.exists():
                record = json.loads(output.read_text(encoding="utf-8"))
                print(f"{keyword_count} keywords, seed {seed} loaded", flush=True)
            elif keyword_count == 500 and config["reuse_500_keyword_baseline"]:
                record = load_500(config, seed)
                output.write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(f"{keyword_count} keywords, seed {seed} reused", flush=True)
            else:
                print(f"{keyword_count} keywords, seed {seed} started", flush=True)
                record = run_new(config, structure_config, keyword_count, seed)
                output.write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(
                    f"{keyword_count} keywords, seed {seed} completed in "
                    f"{record['wall_seconds']:.1f} seconds",
                    flush=True,
                )
            records.append(record)

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
