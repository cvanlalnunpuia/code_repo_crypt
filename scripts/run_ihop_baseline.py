from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "ihop-code"
CONFIG_PATH = ROOT / "configs" / "ihop_enron_baseline.json"
RESULTS = ROOT / "results" / "ihop_enron_baseline"
DATASET = UPSTREAM / "datasets_pro" / "enron-full.pkl"
MPL_CACHE = ROOT / ".cache" / "matplotlib-py39"


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


def package_versions() -> dict[str, str]:
    names = ("matplotlib", "numpy", "scikit-learn", "scipy")
    return {name: importlib.metadata.version(name) for name in names}


def build_params(config: dict):
    sys.path.insert(0, str(UPSTREAM))
    from exp_params import ExpParams

    params = ExpParams()
    params.set_general_params(
        dataset=config["dataset"]["name"],
        nkw=config["dataset"]["keyword_count"],
        nqr=config["queries"]["count"],
        ndoc=config["dataset"]["selected_documents"],
        freq=config["queries"]["frequency_source"],
        mode_ds=config["dataset"]["auxiliary_split"],
        mode_fs="past",
        mode_kw="rand",
        mode_query=config["queries"]["mode"],
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


def run_seed(config: dict, seed: int) -> dict:
    params = build_params(config)
    from experiment import run_experiment
    started = datetime.now(timezone.utc)
    tick = time.perf_counter()
    previous_cwd = Path.cwd()
    try:
        os.chdir(UPSTREAM)
        query_weighted, distinct_macro, upstream_seconds = run_experiment(
            params, seed=seed, debug_mode=True
        )
    finally:
        os.chdir(previous_cwd)
    return {
        "seed": seed,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - tick,
        "upstream_seconds": float(upstream_seconds),
        "query_weighted_recovery": [float(value) for value in query_weighted],
        "distinct_query_macro_recovery": [
            float(value) for value in distinct_macro
        ],
    }


def interval(values: list[float], confidence: float) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    count = len(array)
    mean = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1))
    standard_error = sample_sd / math.sqrt(count)
    critical = float(stats.t.ppf((1 + confidence) / 2, df=count - 1))
    return {
        "n": count,
        "mean": mean,
        "sample_standard_deviation": sample_sd,
        "confidence_level": confidence,
        "confidence_interval_low": mean - critical * standard_error,
        "confidence_interval_high": mean + critical * standard_error,
    }


def aggregate(records: list[dict], config: dict) -> dict:
    checkpoints = config["attack"]["iteration_checkpoints"]
    confidence = config["confidence_level"]
    metrics = {}
    for metric in (
        "query_weighted_recovery",
        "distinct_query_macro_recovery",
    ):
        metrics[metric] = {
            str(checkpoint): interval(
                [record[metric][index] for record in records], confidence
            )
            for index, checkpoint in enumerate(checkpoints)
        }
    return metrics


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    required_hash_seed = str(config["python_hash_seed"])
    if os.environ.get("PYTHONHASHSEED") != required_hash_seed:
        raise RuntimeError(
            f"PYTHONHASHSEED must be {required_hash_seed} before Python starts"
        )
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)
    RESULTS.mkdir(parents=True, exist_ok=True)

    records = []
    for seed in config["seeds"]:
        seed_path = RESULTS / f"seed-{seed}.json"
        if seed_path.exists():
            record = json.loads(seed_path.read_text(encoding="utf-8"))
            print(f"Seed {seed} loaded from {seed_path.name}", flush=True)
        else:
            print(f"Seed {seed} started", flush=True)
            record = run_seed(config, seed)
            seed_path.write_text(
                json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(
                f"Seed {seed} completed in {record['wall_seconds']:.1f} seconds",
                flush=True,
            )
        records.append(record)

    summary = {
        "experiment_id": config["id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "dataset_sha256": sha256(DATASET),
        "upstream_revision": git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "packages": package_versions(),
        "seed_results": records,
        "aggregate_metrics": aggregate(records, config),
    }
    output = RESULTS / "aggregate.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Aggregate written to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
