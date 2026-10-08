from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "ihop-code"
CONFIG_PATH = ROOT / "configs" / "ihop_enron_smoke.json"
RESULTS = ROOT / "results"
DATASET = UPSTREAM / "datasets_pro" / "enron-full.pkl"
MPL_CACHE = ROOT / ".cache" / "matplotlib"


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


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    hash_seed = os.environ.get("PYTHONHASHSEED", "random")
    runtime_id = f"py{sys.version_info.major}{sys.version_info.minor}-hash{hash_seed}"
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)
    sys.path.insert(0, str(UPSTREAM))
    from exp_params import ExpParams
    from experiment import run_experiment

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
    params.set_defense_params("none")
    params.set_attack_params("ihop", mode="Vol", niters=1000, pfree=0.25)
    params.att_params["niter_list"] = config["attack"]["iteration_checkpoints"]

    start = datetime.now(timezone.utc)
    tick = time.perf_counter()
    previous_cwd = Path.cwd()
    try:
        os.chdir(UPSTREAM)
        recovery, accuracy, _ = run_experiment(
            params, seed=config["seed"], debug_mode=True
        )
    finally:
        os.chdir(previous_cwd)
    elapsed = time.perf_counter() - tick

    observed = [float(value) for value in recovery]
    expected = config["expected_recovery"]
    tolerance = config["absolute_tolerance"]
    passed = all(abs(a - b) <= tolerance for a, b in zip(observed, expected))

    result = {
        "experiment_id": config["id"],
        "runtime_id": runtime_id,
        "python_hash_seed": hash_seed,
        "status": "passed" if passed else "failed",
        "started_at": start.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "seed": config["seed"],
        "expected_recovery": expected,
        "observed_recovery": observed,
        "observed_accuracy": [float(value) for value in accuracy],
        "absolute_tolerance": tolerance,
        "dataset_sha256": sha256(DATASET),
        "upstream_revision": git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "configuration": config,
    }
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / f"{config['id']}-{runtime_id}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
