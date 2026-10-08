from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "results" / "access-collision-sweep" / "aggregate.json"
OUTPUT = ROOT / "docs" / "enron-access-collision-sweep.md"


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
    pair_counts = config["forced_pair_counts"]
    checkpoints = config["attack"]["iteration_checkpoints"]

    recovery_rows = []
    structure_rows = []
    for pair_count in pair_counts:
        condition = aggregate["by_condition"][str(pair_count)]
        proportion = condition["forced_keyword_proportion"]
        recovery = condition["recovery"]
        recovery_rows.append(
            "| {pairs} | {proportion:.1%} | {initial} | {middle} | {final} |".format(
                pairs=pair_count,
                proportion=proportion,
                initial=interval(recovery["0"]),
                middle=interval(recovery["100"]),
                final=interval(recovery["1000"]),
            )
        )
        structure = condition["structure"]
        structure_rows.append(
            "| {pairs} | {identical} | {frequency} | {jaccard} | {density} |".format(
                pairs=pair_count,
                identical=interval(condition["identical_pattern_proportion"]),
                frequency=interval(
                    structure["exact_frequency_alternative_proportion"]
                ),
                jaccard=interval(
                    structure["median_nearest_access_jaccard"]
                ),
                density=interval(structure["matrix_density"]),
            )
        )

    paired_rows = []
    for pair_count in pair_counts[1:]:
        condition = aggregate["by_condition"][str(pair_count)]
        difference = aggregate["paired_differences_from_control"][
            str(pair_count)
        ][str(checkpoints[-1])]
        paired_rows.append(
            "| {pairs} | {proportion:.1%} | {difference} |".format(
                pairs=pair_count,
                proportion=condition["forced_keyword_proportion"],
                difference=interval(difference),
            )
        )

    final_control = aggregate["by_condition"]["0"]["recovery"]["1000"]
    final_high = aggregate["by_condition"][str(pair_counts[-1])]["recovery"][
        "1000"
    ]
    high_difference = aggregate["paired_differences_from_control"][
        str(pair_counts[-1])
    ]["1000"]
    high_records = [
        record
        for record in result["records"]
        if record["forced_pair_count"] == pair_counts[-1]
    ]

    text = f"""# Exact access-pattern collisions reduce IHOP recovery on Enron

The experiment used {len(config['seeds'])} paired seeds, {config['keyword_count']} keywords, {config['client_documents_after_split']:,} client documents and {config['selected_documents'] - config['client_documents_after_split']:,} auxiliary documents. Consecutive selected keyword pairs were given identical access columns in both matrices. The keyword set, document split, query order and attack settings remained fixed within each seed.

## Recovery

Values are means with 95% t confidence intervals across seeds.

| Forced pairs | Keywords in forced pairs | 0 iterations | 100 iterations | 1,000 iterations |
|---:|---:|---:|---:|---:|
{chr(10).join(recovery_rows)}

Mean final recovery was {final_control['mean']:.3f} in the control and {final_high['mean']:.3f} with {pair_counts[-1]} forced pairs. Final recovery under the highest collision level ranged from {min(record['query_weighted_recovery'][-1] for record in high_records):.3f} to {max(record['query_weighted_recovery'][-1] for record in high_records):.3f} across seeds.

| Forced pairs | Keywords in forced pairs | Paired final difference from control |
|---:|---:|---:|
{chr(10).join(paired_rows)}

The mean paired change at the highest collision level was {high_difference['mean']:.3f}. Its 95% confidence interval was [{high_difference['confidence_interval_low']:.3f}, {high_difference['confidence_interval_high']:.3f}].

## Structure

The intervention produced the requested number of exact access-pattern pairs in every client and auxiliary matrix. Values are means with 95% t confidence intervals.

| Forced pairs | Identical-pattern keywords | Exact-frequency alternatives | Nearest access Jaccard | Matrix density |
|---:|---:|---:|---:|---:|
{chr(10).join(structure_rows)}

Exact access-pattern collisions reduced recovery at every nonzero intervention level. The intervention also increased frequency ambiguity because each copied access column had the donor column's frequency. It therefore estimates the effect of exact leakage-signature collisions as a joint structural intervention. A near-similarity intervention is required to separate access similarity from exact frequency equality.

All reported values were generated from `results/access-collision-sweep/aggregate.json` by `scripts/write_access_collision_report.py`.
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
