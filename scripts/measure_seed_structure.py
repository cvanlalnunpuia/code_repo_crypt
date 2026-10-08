from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse, stats


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "ihop-code"
CONFIG_PATH = ROOT / "configs" / "enron_seed_structure.json"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT / "scripts"))

from exp_params import ExpParams
from experiment import generate_train_test_data
from measure_structure import access_similarity, frequency_ambiguity, gini, quantile_map


def canonical_hash(value) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def build_params(config: dict) -> ExpParams:
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
    return params


def restricted_matrix(
    dataset: list[list[int]], selected_keywords: list[int]
) -> sparse.csr_matrix:
    mapping = {keyword: index for index, keyword in enumerate(selected_keywords)}
    rows = []
    columns = []
    for row, document in enumerate(dataset):
        for keyword in document:
            column = mapping.get(keyword)
            if column is not None:
                rows.append(row)
                columns.append(column)
    data = np.ones(len(rows), dtype=np.int32)
    return sparse.csr_matrix(
        (data, (rows, columns)),
        shape=(len(dataset), len(selected_keywords)),
        dtype=np.int32,
    )


def exact_cooccurrence_similarity(
    intersections: np.ndarray,
    tri_i: np.ndarray,
    tri_j: np.ndarray,
    quantiles: list[float],
    thresholds: list[float],
) -> dict:
    profiles = intersections.astype(np.float32, copy=True)
    np.fill_diagonal(profiles, 0)
    norms = np.linalg.norm(profiles, axis=1)
    denominator = norms[:, None] * norms[None, :]
    similarities = np.divide(
        profiles @ profiles.T,
        denominator,
        out=np.zeros(profiles.shape, dtype=np.float32),
        where=denominator != 0,
    )
    pair_values = similarities[tri_i, tri_j]
    return {
        "pair_count": int(len(pair_values)),
        "quantiles": quantile_map(pair_values, quantiles),
        "mean": float(np.mean(pair_values)),
        "sample_standard_deviation": float(np.std(pair_values, ddof=1)),
        "threshold_pair_proportions": {
            f"{threshold:g}": float(np.mean(pair_values >= threshold))
            for threshold in thresholds
        },
    }


def profile_matrix(matrix: sparse.csr_matrix, config: dict) -> dict:
    document_count, keyword_count = matrix.shape
    incidence_count = int(matrix.nnz)
    document_lengths = np.diff(matrix.indptr)
    document_frequency = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int32)
    intersections = (matrix.T @ matrix).toarray().astype(np.int32, copy=False)
    tri_i, tri_j = np.triu_indices(keyword_count, k=1)
    frequency = frequency_ambiguity(
        document_frequency,
        tri_i,
        tri_j,
        config["frequency_ambiguity_tolerances_documents"],
    )
    access = access_similarity(
        intersections,
        document_frequency,
        tri_i,
        tri_j,
        config["quantiles"],
        config["access_similarity_thresholds"],
    )
    cooccurrence = exact_cooccurrence_similarity(
        intersections,
        tri_i,
        tri_j,
        config["quantiles"],
        config["cooccurrence_similarity_thresholds"],
    )
    groups = Counter(document_frequency.tolist())
    probabilities = document_frequency / document_frequency.sum()
    positive_probabilities = probabilities[probabilities > 0]
    entropy = float(
        -np.sum(positive_probabilities * np.log(positive_probabilities))
    )
    document_frequency_cv = float(
        np.std(document_frequency) / np.mean(document_frequency)
    )
    density = incidence_count / (document_count * keyword_count)
    primary = {
        "matrix_density": density,
        "document_frequency_cv": document_frequency_cv,
        "exact_frequency_alternative_proportion": frequency["0"][
            "keywords_with_alternative_proportion"
        ],
        "five_document_frequency_alternative_proportion": frequency["5"][
            "keywords_with_alternative_proportion"
        ],
        "median_nearest_access_jaccard": access["nearest_neighbour_quantiles"][
            "0.5"
        ],
        "access_pair_proportion_ge_0_5": access["threshold_pair_proportions"][
            "0.5"
        ],
        "median_cooccurrence_cosine": cooccurrence["quantiles"]["0.5"],
    }
    return {
        "schema": {
            "document_count": document_count,
            "keyword_count": keyword_count,
            "incidence_count": incidence_count,
            "matrix_density": density,
            "empty_document_count": int(np.count_nonzero(document_lengths == 0)),
        },
        "document_length": {
            "mean": float(np.mean(document_lengths)),
            "sample_standard_deviation": float(np.std(document_lengths, ddof=1)),
            "quantiles": quantile_map(document_lengths, config["quantiles"]),
        },
        "document_frequency": {
            "mean": float(np.mean(document_frequency)),
            "sample_standard_deviation": float(
                np.std(document_frequency, ddof=1)
            ),
            "quantiles": quantile_map(document_frequency, config["quantiles"]),
            "population_coefficient_of_variation": document_frequency_cv,
            "gini_coefficient": gini(document_frequency),
            "normalised_entropy": entropy / math.log(keyword_count),
        },
        "frequency_ambiguity": frequency,
        "exact_frequency_groups": {
            "non_singleton_group_count": sum(
                1 for size in groups.values() if size > 1
            ),
            "largest_group_size": max(groups.values()),
        },
        "access_pattern_similarity": access,
        "cooccurrence_profile_similarity": cooccurrence,
        "primary_features": primary,
    }


def correlation(x: list[float], y: list[float]) -> dict:
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    return {
        "n": len(x),
        "pearson_r": float(pearson_r),
        "pearson_p_two_sided": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_two_sided": float(spearman_p),
    }


def main() -> None:
    measurement_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    baseline_config = json.loads(
        (ROOT / measurement_config["baseline_configuration"]).read_text(
            encoding="utf-8"
        )
    )
    required_hash_seed = str(baseline_config["python_hash_seed"])
    if os.environ.get("PYTHONHASHSEED") != required_hash_seed:
        raise RuntimeError(
            f"PYTHONHASHSEED must be {required_hash_seed} before Python starts"
        )
    output_directory = ROOT / measurement_config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    params = build_params(baseline_config)

    joined = []
    for seed in baseline_config["seeds"]:
        np.random.seed(seed)
        previous_cwd = Path.cwd()
        try:
            os.chdir(UPSTREAM)
            auxiliary, client, _ = generate_train_test_data(params.gen_params)
        finally:
            os.chdir(previous_cwd)
        selected_keywords = [int(value) for value in client["keywords"]]
        matrix = restricted_matrix(client["dataset"], selected_keywords)
        structure = profile_matrix(matrix, measurement_config)
        baseline_path = (
            ROOT
            / measurement_config["baseline_results"]
            / f"seed-{seed}.json"
        )
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        record = {
            "seed": seed,
            "selected_keyword_ids": selected_keywords,
            "selected_keyword_ids_sha256": canonical_hash(selected_keywords),
            "client_documents_sha256": canonical_hash(client["dataset"]),
            "auxiliary_documents_sha256": canonical_hash(auxiliary["dataset"]),
            "structure": structure,
            "query_weighted_recovery": baseline["query_weighted_recovery"],
            "distinct_query_macro_recovery": baseline[
                "distinct_query_macro_recovery"
            ],
        }
        (output_directory / f"seed-{seed}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        joined.append(record)
        print(f"Seed {seed} measured", flush=True)

    checkpoints = baseline_config["attack"]["iteration_checkpoints"]
    correlations = {}
    for feature in measurement_config["primary_features"]:
        x = [record["structure"]["primary_features"][feature] for record in joined]
        correlations[feature] = {
            str(checkpoint): correlation(
                x,
                [
                    record["query_weighted_recovery"][index]
                    for record in joined
                ],
            )
            for index, checkpoint in enumerate(checkpoints)
        }

    aggregate = {
        "analysis_id": measurement_config["id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": measurement_config["analysis_scope"],
        "measurement_configuration": measurement_config,
        "baseline_configuration": baseline_config,
        "seed_count": len(joined),
        "seed_records": joined,
        "correlations": correlations,
    }
    (output_directory / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(output_directory / "aggregate.json")


if __name__ == "__main__":
    main()
