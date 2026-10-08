from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "ihop-code"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_volume_support(documents: list[list[int]], keyword_count: int, label: str) -> None:
    for keyword in range(keyword_count):
        count = sum(keyword in document for document in documents)
        if count == 0 or count == len(documents):
            raise ValueError(f"{label} has boundary access probability for keyword {keyword}")


def run_dataset(path: Path, seed: int, iterations: int, checkpoints: list[int], free_fraction: float, attack_mode: str) -> dict[str, Any]:
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from defense import generate_observations
        from exp_params import ExpParams
        from experiment import run_attack

    with path.open("rb") as stream:
        dataset, keywords, metadata = pickle.load(stream)
    fixed = metadata["fixed_split"]
    auxiliary_documents = [dataset[index] for index in fixed["auxiliary_indices"]]
    client_documents = [dataset[index] for index in fixed["client_indices"]]
    if not auxiliary_documents or not client_documents:
        raise ValueError(f"Dataset {path} has an empty fixed split")
    keyword_ids = list(range(len(keywords)))
    if int(free_fraction * len(keyword_ids)) <= 1:
        raise ValueError(f"Dataset {path} has too few keywords for the free fraction")
    if attack_mode == "Vol":
        validate_volume_support(auxiliary_documents, len(keywords), "auxiliary split")
        validate_volume_support(client_documents, len(keywords), "client split")
    auxiliary = {
        "dataset": auxiliary_documents,
        "keywords": keyword_ids,
        "frequencies": np.ones(len(keywords), dtype=float) / len(keywords),
        "mode_query": "each",
    }
    client = {
        "dataset": client_documents,
        "keywords": keyword_ids,
        "frequencies": None,
    }
    params = ExpParams()
    params.set_defense_params("none")
    params.set_attack_params("ihop", mode=attack_mode, niters=iterations, pfree=free_fraction)
    params.att_params["niter_list"] = checkpoints
    real_queries = keyword_ids
    np.random.seed(seed)
    observations, _, evaluated_queries = generate_observations(client, params.def_params, real_queries)
    with contextlib.redirect_stdout(io.StringIO()):
        predictions_at_checkpoints = run_attack("ihop", obs=observations, aux=auxiliary, exp_params=params)
    recovery = [
        sum(int(predicted == query) for predicted, query in zip(predictions, evaluated_queries)) / len(evaluated_queries)
        for predictions in predictions_at_checkpoints
    ]
    return {
        "dataset_sha256": file_sha256(path),
        "document_count": len(dataset),
        "auxiliary_document_count": len(auxiliary_documents),
        "client_document_count": len(client_documents),
        "keyword_count": len(keywords),
        "checkpoints": checkpoints,
        "recovery": recovery,
    }


def run_smoke(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    manifest_path = ROOT / config["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = {}
    for dataset_id, item in sorted(manifest["datasets"].items()):
        results[dataset_id] = run_dataset(
            manifest_path.parent / item["path"],
            int(config["seed"]),
            int(config["iterations"]),
            [int(value) for value in config["iteration_checkpoints"]],
            float(config["free_fraction"]),
            config["attack_mode"],
        )
    return {
        "status": "executed",
        "config": config,
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "upstream_revision": (ROOT / "vendor" / "ihop-code.commit").read_text(encoding="utf-8").strip(),
        "datasets": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an IHOP format and execution smoke check on corpus exports.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run_smoke(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Executed IHOP smoke checks for {len(result['datasets'])} datasets")


if __name__ == "__main__":
    main()
