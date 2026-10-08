from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = (
    ROOT
    / "results"
    / "structure"
    / "enron-baseline-seeds"
    / "aggregate.json"
)
OUTPUT = ROOT / "docs" / "enron-seed-structure-recovery.md"


LABELS = {
    "matrix_density": "Matrix density",
    "document_frequency_cv": "Document-frequency CV",
    "exact_frequency_alternative_proportion": "Exact-frequency alternative proportion",
    "five_document_frequency_alternative_proportion": "Five-document alternative proportion",
    "median_nearest_access_jaccard": "Median nearest access Jaccard",
    "access_pair_proportion_ge_0_5": "Access-pair proportion at Jaccard 0.5",
    "median_cooccurrence_cosine": "Median co-occurrence cosine",
}


def main() -> None:
    result = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    records = result["seed_records"]
    features = result["measurement_configuration"]["primary_features"]
    checkpoint = str(
        max(result["baseline_configuration"]["attack"]["iteration_checkpoints"])
    )

    rows = []
    for feature in features:
        values = [
            record["structure"]["primary_features"][feature]
            for record in records
        ]
        correlation = result["correlations"][feature][checkpoint]
        rows.append(
            "| {label} | {minimum:.6f} | {maximum:.6f} | {pearson:.3f} | {pearson_p:.3f} | {spearman:.3f} | {spearman_p:.3f} |".format(
                label=LABELS[feature],
                minimum=min(values),
                maximum=max(values),
                pearson=correlation["pearson_r"],
                pearson_p=correlation["pearson_p_two_sided"],
                spearman=correlation["spearman_rho"],
                spearman_p=correlation["spearman_p_two_sided"],
            )
        )

    final_recovery = np.asarray(
        [record["query_weighted_recovery"][-1] for record in records]
    )
    strongest = max(
        features,
        key=lambda feature: abs(
            result["correlations"][feature][checkpoint]["spearman_rho"]
        ),
    )
    strongest_result = result["correlations"][strongest][checkpoint]

    text = f"""# Enron seed structure and recovery

The analysis joined {len(records)} seed-specific client matrices to their recorded IHOP recovery. Each matrix contains {records[0]['structure']['schema']['document_count']:,} documents and {records[0]['structure']['schema']['keyword_count']:,} selected keywords. Mean recovery after {int(checkpoint):,} iterations was {np.mean(final_recovery):.3f}, with a range from {np.min(final_recovery):.3f} to {np.max(final_recovery):.3f}.

| Structural feature | Minimum | Maximum | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

{LABELS[strongest]} had the largest absolute Spearman association with final recovery. Spearman rho was {strongest_result['spearman_rho']:.3f}, and its two-sided p value was {strongest_result['spearman_p_two_sided']:.3f}. The analysis is exploratory. Ten observations provide limited precision, final recovery has a ceiling near one, and p values are unadjusted across features and checkpoints.

Each seed record contains the selected keyword identifiers, keyword-set digest, client-document digest, auxiliary-document digest, structural measurements and recovery checkpoints. All values in this report were generated from `results/structure/enron-baseline-seeds/aggregate.json` by `scripts/write_seed_structure_report.py`.
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
