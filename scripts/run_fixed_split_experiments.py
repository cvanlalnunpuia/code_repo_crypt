from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import pickle
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from run_ihop_corpus_smoke import ROOT, UPSTREAM, file_sha256, validate_volume_support


def generate_queries(distribution: str, keyword_count: int, multiplier: int) -> list[int]:
    if distribution == "each":
        return [int(value) for value in np.random.permutation(keyword_count)]
    query_count = keyword_count * multiplier
    if distribution == "uniform_iid":
        probabilities = np.ones(keyword_count, dtype=float) / keyword_count
    elif distribution == "zipf_iid":
        probabilities = 1 / np.arange(1, keyword_count + 1, dtype=float)
        probabilities /= probabilities.sum()
    else:
        raise ValueError(f"Unknown query distribution {distribution}")
    return [int(value) for value in np.random.choice(keyword_count, size=query_count, p=probabilities)]


def recovery_scores(predictions: list[int], queries: list[int]) -> dict[str, float]:
    correct = [int(prediction == query) for prediction, query in zip(predictions, queries)]
    by_keyword: dict[int, list[int]] = defaultdict(list)
    for query, value in zip(queries, correct):
        by_keyword[query].append(value)
    return {
        "query_weighted_recovery": float(np.mean(correct)),
        "distinct_keyword_macro_recovery": float(np.mean([np.mean(values) for values in by_keyword.values()])),
        "observed_distinct_keyword_count": len(by_keyword),
    }


def run_one(path: Path, dataset_id: str, config: dict[str, Any], distribution: str, seed: int) -> dict[str, Any]:
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
        raise ValueError(f"Dataset {dataset_id} has an empty split")
    keyword_count = len(keywords)
    if int(float(config["free_fraction"]) * keyword_count) <= 1:
        raise ValueError(f"Dataset {dataset_id} has too few keywords")
    if config["attack_mode"] == "Vol":
        validate_volume_support(auxiliary_documents, keyword_count, f"{dataset_id} auxiliary split")
        validate_volume_support(client_documents, keyword_count, f"{dataset_id} client split")
    keyword_ids = list(range(keyword_count))
    auxiliary = {
        "dataset": auxiliary_documents,
        "keywords": keyword_ids,
        "frequencies": np.ones(keyword_count, dtype=float) / keyword_count,
        "mode_query": "each" if distribution == "each" else "iid",
    }
    client = {"dataset": client_documents, "keywords": keyword_ids, "frequencies": None}
    params = ExpParams()
    params.set_defense_params("none")
    params.set_attack_params(
        "ihop",
        mode=config["attack_mode"],
        niters=int(config["iterations"]),
        pfree=float(config["free_fraction"]),
    )
    checkpoints = [int(value) for value in config["iteration_checkpoints"]]
    params.att_params["niter_list"] = checkpoints
    np.random.seed(seed)
    queries = generate_queries(distribution, keyword_count, int(config["query_count_per_keyword"]))
    observations, _, evaluated_queries = generate_observations(client, params.def_params, queries)
    with contextlib.redirect_stdout(io.StringIO()):
        predictions = run_attack("ihop", obs=observations, aux=auxiliary, exp_params=params)
    scores = [recovery_scores(values, evaluated_queries) for values in predictions]
    return {
        "dataset_id": dataset_id,
        "dataset_sha256": file_sha256(path),
        "distribution": distribution,
        "seed": seed,
        "query_count": len(queries),
        "keyword_count": keyword_count,
        "checkpoints": checkpoints,
        "scores": scores,
    }


def summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], dict[str, list[float]]] = {}
    for run in runs:
        for checkpoint, scores in zip(run["checkpoints"], run["scores"]):
            key = (run["dataset_id"], run["distribution"], checkpoint)
            bucket = grouped.setdefault(key, {"query_weighted_recovery": [], "distinct_keyword_macro_recovery": []})
            for metric in bucket:
                bucket[metric].append(scores[metric])
    output = {}
    for (dataset_id, distribution, checkpoint), metrics in sorted(grouped.items()):
        output.setdefault(dataset_id, {}).setdefault(distribution, {})[str(checkpoint)] = {
            metric: summary(values) for metric, values in metrics.items()
        }
    return output


def run_task(task: tuple[Path, str, dict[str, Any], str, int]) -> dict[str, Any]:
    return run_one(*task)


def run_experiments(config_path: Path, workers: int = 1, progress: bool = False, checkpoint_path: Path | None = None) -> dict[str, Any]:
    if workers < 1:
        raise ValueError('workers must be positive')
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    if sorted(set(config["iteration_checkpoints"])) != sorted(config["iteration_checkpoints"]):
        raise ValueError("Iteration checkpoints must be unique")
    if max(config["iteration_checkpoints"]) > int(config["iterations"]):
        raise ValueError("Iteration checkpoint exceeds iterations")
    manifest_path = ROOT / config["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = []
    for dataset_id, item in sorted(manifest["datasets"].items()):
        for distribution in config["query_distributions"]:
            for seed in config["seeds"]:
                tasks.append((manifest_path.parent / item['path'], dataset_id, config, distribution, int(seed)))
    identity = {
        'config_sha256': file_sha256(config_path),
        'manifest_sha256': file_sha256(manifest_path),
        'upstream_revision': (ROOT / 'vendor' / 'ihop-code.commit').read_text(encoding='utf-8').strip(),
    }
    cached = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        if checkpoint['identity'] != identity:
            raise ValueError('Checkpoint provenance differs from requested experiment')
        for result in checkpoint['runs']:
            key = (result['dataset_id'], result['distribution'], result['seed'])
            if key in cached:
                raise ValueError('Checkpoint contains duplicate runs')
            cached[key] = result
        expected = {(task[1], task[3], task[4]) for task in tasks}
        if not set(cached) <= expected:
            raise ValueError('Checkpoint contains unexpected runs')
    remaining = [task for task in tasks if (task[1], task[3], task[4]) not in cached]
    runs = list(cached.values())
    with contextlib.ExitStack() as stack:
        if workers == 1:
            results = map(run_task, remaining)
        else:
            executor = stack.enter_context(ProcessPoolExecutor(max_workers=workers))
            results = executor.map(run_task, remaining)
        for result in results:
            runs.append(result)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
                temporary.write_text(json.dumps({'identity': identity, 'runs': runs}, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
                temporary.replace(checkpoint_path)
            if progress:
                print(f"Completed {len(runs)}/{len(tasks)} {result['dataset_id']} {result['distribution']} seed {result['seed']}", flush=True)
    by_key = {(run['dataset_id'], run['distribution'], run['seed']): run for run in runs}
    runs = [by_key[(task[1], task[3], task[4])] for task in tasks]
    return {
        "status": "executed",
        "config": config,
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "upstream_revision": (ROOT / "vendor" / "ihop-code.commit").read_text(encoding="utf-8").strip(),
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IHOP against fixed bilingual corpus splits.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    result = run_experiments(args.config, workers=args.workers, progress=args.progress, checkpoint_path=args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Executed {len(result['runs'])} fixed-split IHOP runs")


if __name__ == "__main__":
    main()
