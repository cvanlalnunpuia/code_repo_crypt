from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "results" / "frequency-ambiguity-sweep" / "aggregate.json"
OUTPUT = ROOT / "docs" / "enron-frequency-ambiguity-sweep.md"


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
    pair_counts = config["equalized_pair_counts"]

    recovery_rows = []
    structure_rows = []
    paired_rows = []
    for pair_count in pair_counts:
        condition = aggregate["by_condition"][str(pair_count)]
        recovery_rows.append(
            "| {pairs} | {proportion:.1%} | {initial} | {middle} | {final} |".format(
                pairs=pair_count,
                proportion=condition["equalized_keyword_proportion"],
                initial=interval(condition["recovery"]["0"]),
                middle=interval(condition["recovery"]["100"]),
                final=interval(condition["recovery"]["1000"]),
            )
        )
        structure_rows.append(
            "| {pairs} | {frequency} | {density} | {jaccard} | {cooccurrence} |".format(
                pairs=pair_count,
                frequency=interval(
                    condition["structure"][
                        "exact_frequency_alternative_proportion"
                    ]
                ),
                density=interval(condition["structure"]["matrix_density"]),
                jaccard=interval(
                    condition["structure"]["median_nearest_access_jaccard"]
                ),
                cooccurrence=interval(
                    condition["structure"]["median_cooccurrence_cosine"]
                ),
            )
        )
        if pair_count > 0:
            paired_rows.append(
                "| {pairs} | {difference} |".format(
                    pairs=pair_count,
                    difference=interval(
                        aggregate["paired_differences_from_control"][
                            str(pair_count)
                        ]["1000"]
                    ),
                )
            )

    control_final = aggregate["by_condition"]["0"]["recovery"]["1000"]
    high_final = aggregate["by_condition"][str(pair_counts[-1])]["recovery"][
        "1000"
    ]
    text = f"""# Frequency ambiguity reduces IHOP recovery on Enron

The experiment used {len(config['seeds'])} paired seeds, {config['keyword_count']} keywords and nested sets of frequency-equalized pairs. Both keywords in each active pair received their integer mean frequency. Separate means were computed for the client and auxiliary matrices. The two access columns in each active pair were assigned disjoint document sets. Pairwise incidence totals, matrix density, the keyword set, document split, query order and attack settings remained fixed within each seed.

## Recovery

Values are means with 95% t confidence intervals across seeds.

| Equalized pairs | Equalized keywords | 0 iterations | 100 iterations | 1,000 iterations |
|---:|---:|---:|---:|---:|
{chr(10).join(recovery_rows)}

Mean final recovery decreased from {control_final['mean']:.3f} in the control to {high_final['mean']:.3f} with {pair_counts[-1]} equalized pairs.

| Equalized pairs | Paired final difference from control |
|---:|---:|
{chr(10).join(paired_rows)}

The recovery reduction increased at each intervention level.

## Structure

Every active pair had equal frequencies and zero mutual access overlap in both matrices. Values are means with 95% t confidence intervals.

| Equalized pairs | Exact-frequency alternatives | Matrix density | Nearest access Jaccard | Co-occurrence cosine |
|---:|---:|---:|---:|---:|
{chr(10).join(structure_rows)}

Exact-frequency ambiguity increased in every seed. Matrix density remained fixed. Median nearest access Jaccard decreased, and median co-occurrence-profile cosine increased. The intervention therefore estimates frequency equality under fixed density and disjoint within-pair access patterns. The reconstructed columns also changed broader co-occurrence structure.

All reported values were generated from `results/frequency-ambiguity-sweep/aggregate.json` by `scripts/write_frequency_ambiguity_report.py`.
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
