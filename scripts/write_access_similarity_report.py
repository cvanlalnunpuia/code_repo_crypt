from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "results" / "access-similarity-sweep" / "aggregate.json"
OUTPUT = ROOT / "docs" / "enron-access-similarity-sweep.md"


def interval(summary: dict, digits: int = 3) -> str:
    return (
        f"{summary['mean']:.{digits}f} "
        f"[{summary['confidence_interval_low']:.{digits}f}, "
        f"{summary['confidence_interval_high']:.{digits}f}]"
    )


def main() -> None:
    result = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    config = result["configuration"]
    aggregate = result["aggregate"]
    conditions = ["control"] + [
        f"target_{target:.2f}" for target in config["target_jaccard_values"]
    ]

    recovery_rows = []
    structure_rows = []
    paired_rows = []
    for condition in conditions:
        summary = aggregate["by_condition"][condition]
        label = "Control" if condition == "control" else f"{summary['target_jaccard']:.2f}"
        recovery_rows.append(
            "| {label} | {initial} | {middle} | {final} |".format(
                label=label,
                initial=interval(summary["recovery"]["0"]),
                middle=interval(summary["recovery"]["100"]),
                final=interval(summary["recovery"]["1000"]),
            )
        )
        structure_rows.append(
            "| {label} | {client} | {auxiliary} | {frequency} | {density} | {cooccurrence} |".format(
                label=label,
                client=interval(summary["client_pair_jaccard"]),
                auxiliary=interval(summary["auxiliary_pair_jaccard"]),
                frequency=interval(
                    summary["structure"][
                        "exact_frequency_alternative_proportion"
                    ]
                ),
                density=interval(summary["structure"]["matrix_density"]),
                cooccurrence=interval(
                    summary["structure"]["median_cooccurrence_cosine"]
                ),
            )
        )
        if condition != "control":
            paired_rows.append(
                "| {label} | {difference} |".format(
                    label=label,
                    difference=interval(
                        aggregate["paired_differences_from_control"][condition][
                            "1000"
                        ]
                    ),
                )
            )

    text = f"""# Frequency-preserving access similarity lowers IHOP recovery on Enron

The experiment used {len(config['seeds'])} paired seeds, {config['keyword_count']} keywords and {config['pair_count']} disjoint keyword pairs per seed. Recipient columns were reconstructed at target Jaccard similarities of 0.25, 0.50 and 0.75. Each recipient retained its original document frequency in the client and auxiliary matrices. The keyword set, document split, query order, matrix density and attack settings remained fixed within each seed.

## Recovery

Values are means with 95% t confidence intervals across seeds.

| Target pair Jaccard | 0 iterations | 100 iterations | 1,000 iterations |
|---:|---:|---:|---:|
{chr(10).join(recovery_rows)}

| Target pair Jaccard | Paired final difference from control |
|---:|---:|
{chr(10).join(paired_rows)}

Every intervention condition had lower mean final recovery than its paired control. The reductions were 0.031 at target 0.25, 0.021 at target 0.50 and 0.035 at target 0.75. The three recovery means did not form a monotonic sequence.

## Structure

The selected-pair Jaccard values reached their targets within the integer resolution of the access columns. Values are means with 95% t confidence intervals.

| Target | Client pair Jaccard | Auxiliary pair Jaccard | Exact-frequency alternatives | Matrix density | Co-occurrence cosine |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(structure_rows)}

Every client and auxiliary column retained its control frequency. Exact-frequency ambiguity and matrix density were unchanged within each seed. Reassigning document membership changed co-occurrence profiles. The experiment therefore estimates the effect of access-layout similarity under fixed frequency marginals. It does not isolate access similarity from co-occurrence structure.

All reported values were generated from `results/access-similarity-sweep/aggregate.json` by `scripts/write_access_similarity_report.py`.
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
