from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "enron_ablation_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def manipulated_measure(contrast_id: str, source: dict) -> dict:
    aggregate = source["aggregate"]
    if contrast_id == "keyword_size":
        return {
            "label": "Keyword count",
            "baseline": 100.0,
            "intervention": 500.0,
        }
    if contrast_id == "exact_collisions":
        conditions = aggregate["by_condition"]
        return {
            "label": "Identical-pattern keyword proportion",
            "baseline": conditions["0"]["identical_pattern_proportion"]["mean"],
            "intervention": conditions["62"]["identical_pattern_proportion"]["mean"],
        }
    if contrast_id == "access_similarity":
        conditions = aggregate["by_condition"]
        return {
            "label": "Selected-pair client Jaccard",
            "baseline": conditions["control"]["client_pair_jaccard"]["mean"],
            "intervention": conditions["target_0.75"]["client_pair_jaccard"]["mean"],
        }
    if contrast_id == "frequency_ambiguity":
        conditions = aggregate["by_condition"]
        return {
            "label": "Exact-frequency alternative proportion",
            "baseline": conditions["0"]["structure"]["exact_frequency_alternative_proportion"]["mean"],
            "intervention": conditions["62"]["structure"]["exact_frequency_alternative_proportion"]["mean"],
        }
    if contrast_id == "cooccurrence_similarity":
        conditions = aggregate["by_condition"]
        return {
            "label": "Selected-pair client cosine",
            "baseline": conditions["budget_1"]["client_pair_cooccurrence_cosine"]["mean"],
            "intervention": conditions["budget_128"]["client_pair_cooccurrence_cosine"]["mean"],
        }
    if contrast_id == "sparsity":
        conditions = aggregate["by_condition"]
        return {
            "label": "Matrix density",
            "baseline": conditions["1.00"]["structure"]["matrix_density"]["mean"],
            "intervention": conditions["0.25"]["structure"]["matrix_density"]["mean"],
        }
    raise ValueError(f"Unknown contrast {contrast_id}")


def build_contrast(definition: dict, config: dict) -> dict:
    source_path = ROOT / definition["source"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    field = definition["selector_field"]
    baseline_records = [
        record for record in source["records"] if record[field] == definition["baseline"]
    ]
    intervention_records = [
        record
        for record in source["records"]
        if record[field] == definition["intervention"]
    ]
    baseline_map = {record["seed"]: record for record in baseline_records}
    intervention_map = {record["seed"]: record for record in intervention_records}
    seeds = sorted(set(baseline_map) & set(intervention_map))
    checkpoints = config["checkpoint_iterations"]
    confidence = config["confidence_level"]
    checkpoint_summaries = {}
    for index, checkpoint in enumerate(checkpoints):
        baseline_values = [
            baseline_map[seed]["query_weighted_recovery"][index] for seed in seeds
        ]
        intervention_values = [
            intervention_map[seed]["query_weighted_recovery"][index]
            for seed in seeds
        ]
        differences = [
            intervention - baseline
            for baseline, intervention in zip(baseline_values, intervention_values)
        ]
        checkpoint_summaries[str(checkpoint)] = {
            "baseline_recovery": summary(baseline_values, confidence),
            "intervention_recovery": summary(intervention_values, confidence),
            "paired_difference": summary(differences, confidence),
        }

    return {
        "id": definition["id"],
        "label": definition["label"],
        "contrast_label": definition["contrast_label"],
        "source": definition["source"],
        "source_sha256": sha256(source_path),
        "selector_field": field,
        "baseline_selector": definition["baseline"],
        "intervention_selector": definition["intervention"],
        "seeds": seeds,
        "manipulated_measure": manipulated_measure(definition["id"], source),
        "checkpoints": checkpoint_summaries,
        "final_seed_differences": {
            str(seed): intervention_map[seed]["query_weighted_recovery"][-1]
            - baseline_map[seed]["query_weighted_recovery"][-1]
            for seed in seeds
        },
    }


def interval(value: dict, digits: int = 3) -> str:
    return (
        f"{value['mean']:.{digits}f} "
        f"[{value['confidence_interval_low']:.{digits}f}, "
        f"{value['confidence_interval_high']:.{digits}f}]"
    )


def write_report(result: dict, output: Path) -> None:
    rows = []
    for contrast in result["contrasts"]:
        measure = contrast["manipulated_measure"]
        if measure["label"] == "Keyword count":
            measure_values = (
                f"{measure['baseline']:.0f} to {measure['intervention']:.0f}"
            )
        else:
            measure_values = (
                f"{measure['baseline']:.3f} to {measure['intervention']:.3f}"
            )
        rows.append(
            "| {label} | {contrast_label} | {measure_label}, {measure_values} | {at_100} | {at_1000} |".format(
                label=contrast["label"],
                contrast_label=contrast["contrast_label"],
                measure_label=measure["label"],
                measure_values=measure_values,
                at_100=interval(
                    contrast["checkpoints"]["100"]["paired_difference"]
                ),
                at_1000=interval(
                    contrast["checkpoints"]["1000"]["paired_difference"]
                ),
            )
        )

    by_id = {contrast["id"]: contrast for contrast in result["contrasts"]}
    size_100 = by_id["keyword_size"]["checkpoints"]["100"]["paired_difference"]
    size_final = by_id["keyword_size"]["checkpoints"]["1000"]["paired_difference"]
    coocc_final = by_id["cooccurrence_similarity"]["checkpoints"]["1000"]["paired_difference"]
    text = f"""# Paired endpoint contrasts across Enron structural ablations

Six controlled ablations were compared on the same final-recovery scale. Each row uses the maximum configured intervention level and ten paired seeds. Intervention scales differ across rows, so the table describes endpoint contrasts. Direct comparison as regression coefficients would be invalid.

| Intervention | Endpoint contrast | Manipulated measure | Difference at 100 iterations | Difference at 1,000 iterations |
|---|---|---|---:|---:|
{chr(10).join(rows)}

![Paired final recovery differences](../figures/enron-ablation-final-recovery.png)

Figure 1. Paired differences in final IHOP recovery for the maximum configured endpoint in each Enron ablation.

Incidence retention produced the largest negative endpoint contrast. Frequency equality and exact access collisions followed. Frequency-preserving access similarity produced a smaller negative contrast. The keyword-count contrast was {size_100['mean']:.3f} at 100 iterations and {size_final['mean']:.3f} at 1,000 iterations. The final co-occurrence contrast was {coocc_final['mean']:.3f}.

The access-collision intervention also changed frequency ambiguity. The frequency-equality intervention changed broader co-occurrence structure. Column thinning increased frequency ambiguity as density fell. The co-occurrence analysis used its randomized reconstruction as the paired control. These design differences limit comparisons of causal magnitude across rows.

Confidence intervals are unadjusted 95% t intervals across paired seeds. All values were generated from the six source aggregates by `scripts/build_enron_ablation_summary.py`.
"""
    output.write_text(text, encoding="utf-8")


def write_figure(result: dict, png: Path, pdf: Path) -> None:
    labels = [contrast["label"] for contrast in result["contrasts"]]
    final = [
        contrast["checkpoints"]["1000"]["paired_difference"]
        for contrast in result["contrasts"]
    ]
    means = np.asarray([value["mean"] for value in final])
    lower = np.asarray([value["confidence_interval_low"] for value in final])
    upper = np.asarray([value["confidence_interval_high"] for value in final])
    errors = np.vstack((means - lower, upper - means))
    positions = np.arange(len(labels))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
        }
    )
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.errorbar(
        means,
        positions,
        xerr=errors,
        fmt="o",
        color="#1f4e79",
        ecolor="#1f4e79",
        elinewidth=1.4,
        capsize=3,
        markersize=5,
    )
    axis.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlim(-0.92, 0.10)
    axis.set_xlabel("Paired difference in final recovery")
    axis.set_title("Paired change in final IHOP recovery")
    axis.grid(axis="x", color="#dddddd", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.subplots_adjust(left=0.30, right=0.97, top=0.88, bottom=0.16)
    figure.savefig(
        png,
        dpi=300,
        metadata={"Software": "Research analysis"},
    )
    figure.savefig(
        pdf,
        metadata={
            "Creator": "Research analysis",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    contrasts = [build_contrast(definition, config) for definition in config["contrasts"]]
    result = {
        "experiment_id": config["id"],
        "confidence_level": config["confidence_level"],
        "checkpoint_iterations": config["checkpoint_iterations"],
        "configuration_sha256": sha256(CONFIG_PATH),
        "contrasts": contrasts,
    }
    output_json = ROOT / config["output_json"]
    output_report = ROOT / config["output_report"]
    output_png = ROOT / config["output_figure_png"]
    output_pdf = ROOT / config["output_figure_pdf"]
    for path in (output_json, output_report, output_png, output_pdf):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(result, output_report)
    write_figure(result, output_png, output_pdf)
    print(output_json)
    print(output_report)
    print(output_png)
    print(output_pdf)


if __name__ == "__main__":
    main()
