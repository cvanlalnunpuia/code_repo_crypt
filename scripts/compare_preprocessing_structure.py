from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: np.ndarray, levels: list[float]) -> dict[str, float | None]:
    if not len(values):
        return {f"{level:g}": None for level in levels}
    computed = np.quantile(values, levels)
    return {f"{level:g}": float(value) for level, value in zip(levels, computed)}


def sample_standard_deviation(values: np.ndarray) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def gini(values: np.ndarray) -> float | None:
    total = float(values.sum())
    if not len(values) or total == 0:
        return None
    ordered = np.sort(values.astype(np.float64))
    count = len(ordered)
    weighted = np.sum(np.arange(1, count + 1) * ordered)
    return float((2 * weighted) / (count * total) - (count + 1) / count)


def profile_condition(condition_dir: Path, language: str, config: dict[str, Any]) -> dict[str, Any]:
    documents = json.loads((condition_dir / "documents.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((condition_dir / "vocabulary.json").read_text(encoding="utf-8"))
    matrix = sparse.load_npz(condition_dir / "incidence.npz").tocsr().astype(np.int32)
    rows = [index for index, document in enumerate(documents) if document["language"] == language]
    if not rows:
        raise ValueError(f"Condition {condition_dir} lacks {language} documents")
    language_matrix = matrix[rows, :]
    full_frequency = np.asarray(language_matrix.sum(axis=0)).ravel().astype(np.int64)
    available = [index for index, value in enumerate(full_frequency) if value > 0]
    available.sort(key=lambda index: (-int(full_frequency[index]), vocabulary[index]["token"]))
    selected = available[: int(config["keyword_universe_size"])]
    selected_tokens = [vocabulary[index]["token"] for index in selected]
    selected_matrix = language_matrix[:, selected].tocsr()
    document_frequency = np.asarray(selected_matrix.sum(axis=0)).ravel().astype(np.int64)
    document_lengths = np.diff(selected_matrix.indptr).astype(np.int64)
    document_count, keyword_count = selected_matrix.shape
    incidence_count = int(selected_matrix.nnz)
    density = incidence_count / (document_count * keyword_count) if document_count and keyword_count else 0.0
    intersections = (selected_matrix.T @ selected_matrix).toarray().astype(np.int64, copy=False)
    tri_i, tri_j = np.triu_indices(keyword_count, k=1)
    pair_count = len(tri_i)

    ambiguity = {}
    gaps = np.abs(document_frequency[tri_i] - document_frequency[tri_j]) if pair_count else np.array([], dtype=np.int64)
    for tolerance in config["frequency_ambiguity_tolerances_documents"]:
        mask = gaps <= tolerance
        ambiguous = np.unique(np.concatenate((tri_i[mask], tri_j[mask]))) if np.any(mask) else np.array([], dtype=int)
        ambiguity[str(tolerance)] = {
            "ambiguous_pair_count": int(np.count_nonzero(mask)),
            "ambiguous_pair_proportion": float(np.mean(mask)) if pair_count else None,
            "keywords_with_alternative_count": int(len(ambiguous)),
            "keywords_with_alternative_proportion": len(ambiguous) / keyword_count if keyword_count else None,
        }

    if pair_count:
        unions = document_frequency[:, None] + document_frequency[None, :] - intersections
        similarity = np.divide(intersections, unions, out=np.zeros(intersections.shape, dtype=float), where=unions != 0)
        np.fill_diagonal(similarity, -1)
        access_pairs = similarity[tri_i, tri_j]
        nearest = np.max(similarity, axis=1)
        identical_mask = access_pairs == 1
        identical_keywords = np.unique(np.concatenate((tri_i[identical_mask], tri_j[identical_mask]))) if np.any(identical_mask) else np.array([], dtype=int)
    else:
        access_pairs = np.array([], dtype=float)
        nearest = np.array([], dtype=float)
        identical_mask = np.array([], dtype=bool)
        identical_keywords = np.array([], dtype=int)

    profiles = intersections.astype(float, copy=True)
    if keyword_count:
        np.fill_diagonal(profiles, 0)
    cooccurrence_sample_size = min(int(config["cooccurrence_pair_sample_size"]), pair_count)
    if cooccurrence_sample_size:
        rng = np.random.default_rng(int(config["cooccurrence_sample_seed"]))
        chosen = rng.choice(pair_count, size=cooccurrence_sample_size, replace=False)
        left = tri_i[chosen]
        right = tri_j[chosen]
        norms = np.linalg.norm(profiles, axis=1)
        denominators = norms[left] * norms[right]
        cooccurrence_values = np.divide(
            np.einsum("ij,ij->i", profiles[left], profiles[right], optimize=True),
            denominators,
            out=np.zeros(cooccurrence_sample_size, dtype=float),
            where=denominators != 0,
        )
    else:
        cooccurrence_values = np.array([], dtype=float)

    frequencies = Counter(document_frequency.tolist())
    group_sizes = [size for size in frequencies.values() if size > 1]
    probability = document_frequency / incidence_count if incidence_count else np.array([], dtype=float)
    entropy = float(-np.sum(probability * np.log(probability))) if len(probability) else None
    normalised_entropy = entropy / math.log(keyword_count) if entropy is not None and keyword_count > 1 else None
    sorted_frequency = np.sort(document_frequency)[::-1]
    top_one = max(1, math.ceil(keyword_count * 0.01)) if keyword_count else 0
    top_ten = max(1, math.ceil(keyword_count * 0.10)) if keyword_count else 0
    mean_frequency = float(np.mean(document_frequency)) if keyword_count else None
    return {
        "artifact_digests": {
            name: file_sha256(condition_dir / name)
            for name in ("documents.json", "vocabulary.json", "incidence.npz", "metadata.json")
        },
        "document_count": document_count,
        "available_keyword_count": len(available),
        "selected_keyword_count": keyword_count,
        "selected_keyword_sha256": hashlib.sha256("\n".join(selected_tokens).encode("utf-8")).hexdigest(),
        "incidence_count": incidence_count,
        "matrix_density": density,
        "document_length": {
            "mean": float(np.mean(document_lengths)),
            "sample_standard_deviation": sample_standard_deviation(document_lengths),
            "quantiles": quantiles(document_lengths, config["quantiles"]),
        },
        "document_frequency": {
            "mean": mean_frequency,
            "sample_standard_deviation": sample_standard_deviation(document_frequency),
            "quantiles": quantiles(document_frequency, config["quantiles"]),
            "population_coefficient_of_variation": float(np.std(document_frequency) / mean_frequency) if mean_frequency else None,
            "gini_coefficient": gini(document_frequency),
            "normalised_entropy": normalised_entropy,
            "top_1_percent_incidence_share": float(sorted_frequency[:top_one].sum() / incidence_count) if incidence_count else None,
            "top_10_percent_incidence_share": float(sorted_frequency[:top_ten].sum() / incidence_count) if incidence_count else None,
        },
        "frequency_ambiguity": ambiguity,
        "exact_frequency_groups": {
            "non_singleton_group_count": len(group_sizes),
            "largest_group_size": max(group_sizes, default=1),
        },
        "access_pattern_similarity": {
            "pair_count": pair_count,
            "all_pair_quantiles": quantiles(access_pairs, config["quantiles"]),
            "positive_pair_proportion": float(np.mean(access_pairs > 0)) if pair_count else None,
            "threshold_pair_proportions": {
                f"{threshold:g}": float(np.mean(access_pairs >= threshold)) if pair_count else None
                for threshold in config["access_similarity_thresholds"]
            },
            "identical_pair_count": int(np.count_nonzero(identical_mask)),
            "keywords_with_identical_pattern_count": int(len(identical_keywords)),
            "keywords_with_identical_pattern_proportion": len(identical_keywords) / keyword_count if keyword_count else None,
            "nearest_neighbour_quantiles": quantiles(nearest, config["quantiles"]),
        },
        "cooccurrence_profile_similarity": {
            "sample_size": cooccurrence_sample_size,
            "sample_seed": int(config["cooccurrence_sample_seed"]),
            "quantiles": quantiles(cooccurrence_values, config["quantiles"]),
            "mean": float(np.mean(cooccurrence_values)) if cooccurrence_sample_size else None,
            "sample_standard_deviation": sample_standard_deviation(cooccurrence_values),
            "threshold_pair_proportions": {
                f"{threshold:g}": float(np.mean(cooccurrence_values >= threshold)) if cooccurrence_sample_size else None
                for threshold in config["cooccurrence_similarity_thresholds"]
            },
        },
    }


def metric_projection(profile: dict[str, Any]) -> dict[str, float | int | None]:
    return {
        "available_keyword_count": profile["available_keyword_count"],
        "selected_keyword_count": profile["selected_keyword_count"],
        "incidence_count": profile["incidence_count"],
        "matrix_density": profile["matrix_density"],
        "mean_document_length": profile["document_length"]["mean"],
        "frequency_cv": profile["document_frequency"]["population_coefficient_of_variation"],
        "exact_frequency_alternative_proportion": profile["frequency_ambiguity"]["0"]["keywords_with_alternative_proportion"],
        "median_nearest_access_jaccard": profile["access_pattern_similarity"]["nearest_neighbour_quantiles"]["0.5"],
        "median_cooccurrence_cosine": profile["cooccurrence_profile_similarity"]["quantiles"]["0.5"],
    }


def difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float | int | None]:
    output = {}
    for key, left_value in metric_projection(left).items():
        right_value = metric_projection(right)[key]
        output[key] = left_value - right_value if left_value is not None and right_value is not None else None
    return output


def build_comparison(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    project_root = config_path.resolve().parent.parent
    profiles = {}
    for condition, relative_path in config["conditions"].items():
        condition_dir = project_root / relative_path
        profiles[condition] = {
            language: profile_condition(condition_dir, language, config)
            for language in config["languages"]
        }
    baseline = "tokenisation_only"
    preprocessing_differences = {
        condition: {
            language: difference(profiles[condition][language], profiles[baseline][language])
            for language in config["languages"]
        }
        for condition in profiles
        if condition != baseline
    }
    left_language, right_language = config["languages"]
    language_differences = {
        condition: difference(language_profiles[left_language], language_profiles[right_language])
        for condition, language_profiles in profiles.items()
    }
    return {
        "config": config,
        "config_sha256": file_sha256(config_path),
        "profiles": profiles,
        "preprocessing_differences_from_tokenisation": preprocessing_differences,
        f"language_differences_{left_language}_minus_{right_language}": language_differences,
    }


def format_value(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def build_report(result: dict[str, Any]) -> str:
    conditions = list(result["profiles"])
    languages = result["config"]["languages"]
    labels = {
        "tokenisation_only": "Tokenisation only",
        "stopword_removal": "Stopword removal",
        "stemming": "Stopwords and stemming",
    }
    lines = [
        "# Preprocessing structural comparison",
        "",
        "The table reports structural measurements generated from each fixture matrix.",
        "",
        "| Condition | Language | Documents | Available keywords | Selected keywords | Incidences | Density | Mean document length | Frequency CV | Exact-frequency alternatives | Median nearest access Jaccard | Median co-occurrence cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        for language in languages:
            profile = result["profiles"][condition][language]
            projected = metric_projection(profile)
            lines.append(
                "| " + " | ".join([
                    labels.get(condition, condition),
                    language,
                    str(profile["document_count"]),
                    str(projected["available_keyword_count"]),
                    str(projected["selected_keyword_count"]),
                    str(projected["incidence_count"]),
                    format_value(projected["matrix_density"]),
                    format_value(projected["mean_document_length"]),
                    format_value(projected["frequency_cv"]),
                    format_value(projected["exact_frequency_alternative_proportion"]),
                    format_value(projected["median_nearest_access_jaccard"]),
                    format_value(projected["median_cooccurrence_cosine"]),
                ]) + " |"
            )
    lines.extend([
        "",
        "The fixture contains one document per language. Its access and co-occurrence similarities are degenerate. The table verifies computation and reporting. Production interpretation requires the matched pilot corpus.",
        "",
        "All values were generated from the condition matrices by `scripts/compare_preprocessing_structure.py`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare leakage-relevant structure across preprocessing conditions.")
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = build_comparison(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(build_report(result), encoding="utf-8")
    print(f"Compared {len(result['profiles'])} preprocessing conditions")


if __name__ == "__main__":
    main()
