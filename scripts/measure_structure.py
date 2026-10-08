from __future__ import annotations

import hashlib
import json
import math
import pickle
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "enron_structural_profile.json"
OUTPUT = ROOT / "results" / "structure" / "enron-full.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile_map(values: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    computed = np.quantile(values, quantiles)
    return {f"{q:g}": float(value) for q, value in zip(quantiles, computed)}


def gini(values: np.ndarray) -> float:
    ordered = np.sort(values.astype(np.float64))
    count = len(ordered)
    weighted_sum = np.sum(np.arange(1, count + 1) * ordered)
    return float((2 * weighted_sum) / (count * ordered.sum()) - (count + 1) / count)


def build_matrix(dataset: list[list[int]], keyword_count: int) -> sparse.csr_matrix:
    row_indices = np.repeat(
        np.arange(len(dataset), dtype=np.int32),
        np.fromiter((len(document) for document in dataset), dtype=np.int32),
    )
    column_indices = np.fromiter(
        (keyword for document in dataset for keyword in document), dtype=np.int32
    )
    data = np.ones(len(column_indices), dtype=np.int32)
    return sparse.csr_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(dataset), keyword_count),
        dtype=np.int32,
    )


def frequency_ambiguity(
    document_frequency: np.ndarray,
    tri_i: np.ndarray,
    tri_j: np.ndarray,
    tolerances: list[int],
) -> dict[str, dict[str, float | int]]:
    gaps = np.abs(document_frequency[tri_i] - document_frequency[tri_j])
    total_pairs = len(gaps)
    output = {}
    for tolerance in tolerances:
        mask = gaps <= tolerance
        pair_count = int(np.count_nonzero(mask))
        ambiguous_keywords = np.unique(
            np.concatenate((tri_i[mask], tri_j[mask]))
        )
        output[str(tolerance)] = {
            "ambiguous_pair_count": pair_count,
            "ambiguous_pair_proportion": pair_count / total_pairs,
            "keywords_with_alternative_count": int(len(ambiguous_keywords)),
            "keywords_with_alternative_proportion": len(ambiguous_keywords)
            / len(document_frequency),
        }
    return output


def access_similarity(
    intersections: np.ndarray,
    document_frequency: np.ndarray,
    tri_i: np.ndarray,
    tri_j: np.ndarray,
    quantiles: list[float],
    thresholds: list[float],
) -> dict:
    unions = (
        document_frequency[:, None]
        + document_frequency[None, :]
        - intersections
    )
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros(intersections.shape, dtype=np.float32),
        where=unions != 0,
    )
    np.fill_diagonal(similarities, -1.0)
    pair_values = similarities[tri_i, tri_j]
    nearest = np.max(similarities, axis=1)
    exact_mask = pair_values == 1.0
    exact_keywords = np.unique(
        np.concatenate((tri_i[exact_mask], tri_j[exact_mask]))
    )
    return {
        "all_pair_quantiles": quantile_map(pair_values, quantiles),
        "positive_pair_proportion": float(np.mean(pair_values > 0)),
        "threshold_pair_proportions": {
            f"{threshold:g}": float(np.mean(pair_values >= threshold))
            for threshold in thresholds
        },
        "identical_pair_count": int(np.count_nonzero(exact_mask)),
        "keywords_with_identical_pattern_count": int(len(exact_keywords)),
        "keywords_with_identical_pattern_proportion": len(exact_keywords)
        / len(document_frequency),
        "nearest_neighbour_quantiles": quantile_map(nearest, quantiles),
    }


def cooccurrence_similarity(
    intersections: np.ndarray,
    tri_i: np.ndarray,
    tri_j: np.ndarray,
    sample_size: int,
    seed: int,
    quantiles: list[float],
    thresholds: list[float],
) -> dict:
    if sample_size > len(tri_i):
        raise ValueError("Sample size exceeds the number of distinct keyword pairs")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(tri_i), size=sample_size, replace=False)
    sample_i = tri_i[selected]
    sample_j = tri_j[selected]

    profiles = intersections.astype(np.float32, copy=True)
    np.fill_diagonal(profiles, 0)
    norms = np.linalg.norm(profiles, axis=1)
    values = np.empty(sample_size, dtype=np.float32)
    chunk_size = 2048
    for start in range(0, sample_size, chunk_size):
        stop = min(start + chunk_size, sample_size)
        left = sample_i[start:stop]
        right = sample_j[start:stop]
        dots = np.einsum(
            "ij,ij->i", profiles[left], profiles[right], optimize=True
        )
        denominators = norms[left] * norms[right]
        values[start:stop] = np.divide(
            dots,
            denominators,
            out=np.zeros(len(left), dtype=np.float32),
            where=denominators != 0,
        )
    return {
        "sample_size": sample_size,
        "sample_seed": seed,
        "quantiles": quantile_map(values, quantiles),
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
        "threshold_pair_proportions": {
            f"{threshold:g}": float(np.mean(values >= threshold))
            for threshold in thresholds
        },
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset_path = ROOT / config["dataset"]
    started = datetime.now(timezone.utc)
    tick = time.perf_counter()
    with dataset_path.open("rb") as stream:
        dataset, keywords, auxiliary = pickle.load(stream)

    matrix = build_matrix(dataset, len(keywords))
    document_lengths = np.diff(matrix.indptr)
    document_frequency = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int32)
    incidence_count = int(matrix.nnz)
    keyword_count = len(keywords)
    document_count = len(dataset)
    probabilities = document_frequency / document_frequency.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    sorted_frequency = np.sort(document_frequency)[::-1]
    top_one_count = max(1, math.ceil(keyword_count * 0.01))
    top_ten_count = max(1, math.ceil(keyword_count * 0.10))

    intersections = (matrix.T @ matrix).toarray().astype(np.int32, copy=False)
    tri_i, tri_j = np.triu_indices(keyword_count, k=1)
    exact_groups = Counter(document_frequency.tolist())
    exact_group_sizes = [size for size in exact_groups.values() if size > 1]

    result = {
        "profile_id": config["id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started.isoformat(),
        "elapsed_seconds": time.perf_counter() - tick,
        "configuration": config,
        "dataset_sha256": sha256(dataset_path),
        "schema": {
            "document_count": document_count,
            "keyword_count": keyword_count,
            "incidence_count": incidence_count,
            "matrix_density": incidence_count / (document_count * keyword_count),
            "auxiliary_keys": sorted(auxiliary.keys()),
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
            "population_coefficient_of_variation": float(
                np.std(document_frequency) / np.mean(document_frequency)
            ),
            "gini_coefficient": gini(document_frequency),
            "normalised_entropy": entropy / math.log(keyword_count),
            "top_1_percent_incidence_share": float(
                sorted_frequency[:top_one_count].sum() / incidence_count
            ),
            "top_10_percent_incidence_share": float(
                sorted_frequency[:top_ten_count].sum() / incidence_count
            ),
        },
        "frequency_ambiguity": frequency_ambiguity(
            document_frequency,
            tri_i,
            tri_j,
            config["frequency_ambiguity_tolerances_documents"],
        ),
        "exact_frequency_groups": {
            "non_singleton_group_count": len(exact_group_sizes),
            "largest_group_size": max(exact_group_sizes, default=1),
        },
        "access_pattern_similarity": access_similarity(
            intersections,
            document_frequency,
            tri_i,
            tri_j,
            config["quantiles"],
            config["access_similarity_thresholds"],
        ),
        "cooccurrence_profile_similarity": cooccurrence_similarity(
            intersections,
            tri_i,
            tri_j,
            config["cooccurrence_pair_sample_size"],
            config["cooccurrence_sample_seed"],
            config["quantiles"],
            config["cooccurrence_similarity_thresholds"],
        ),
    }
    result["elapsed_seconds"] = time.perf_counter() - tick
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
