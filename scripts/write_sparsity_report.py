from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "results" / "sparsity-sweep" / "aggregate.json"
OUTPUT = ROOT / "docs" / "enron-sparsity-sweep.md"


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
    rates = config["retention_rates"]

    recovery_rows = []
    structure_rows = []
    paired_rows = []
    for rate in rates:
        condition = aggregate["by_condition"][f"{rate:.2f}"]
        recovery_rows.append(
            "| {rate:.0%} | {retention} | {initial} | {middle} | {final} |".format(
                rate=rate,
                retention=interval(
                    condition["achieved_client_incidence_retention"]
                ),
                initial=interval(condition["recovery"]["0"]),
                middle=interval(condition["recovery"]["100"]),
                final=interval(condition["recovery"]["1000"]),
            )
        )
        structure_rows.append(
            "| {rate:.0%} | {density} | {frequency} | {jaccard} | {cooccurrence} |".format(
                rate=rate,
                density=interval(condition["structure"]["matrix_density"]),
                frequency=interval(
                    condition["structure"][
                        "exact_frequency_alternative_proportion"
                    ]
                ),
                jaccard=interval(
                    condition["structure"]["median_nearest_access_jaccard"]
                ),
                cooccurrence=interval(
                    condition["structure"]["median_cooccurrence_cosine"]
                ),
            )
        )
        if rate < 1.0:
            paired_rows.append(
                "| {rate:.0%} | {difference} |".format(
                    rate=rate,
                    difference=interval(
                        aggregate["paired_differences_from_control"][
                            f"{rate:.2f}"
                        ]["1000"]
                    ),
                )
            )

    control_final = aggregate["by_condition"]["1.00"]["recovery"]["1000"]
    low_final = aggregate["by_condition"]["0.25"]["recovery"]["1000"]
    text = f"""# Column thinning reduces IHOP recovery on Enron

The experiment used {len(config['seeds'])} paired seeds, {config['keyword_count']} keywords and {config['client_documents_after_split']:,} client documents. Each keyword retained a nested deterministic subset of its original access set. Retention rates were 0.75, 0.50 and 0.25. Document count, keyword count, query order and weak frequency ordering remained fixed. Every keyword retained at least one incidence.

## Recovery

Values are means with 95% t confidence intervals across seeds.

| Configured retention | Achieved client retention | 0 iterations | 100 iterations | 1,000 iterations |
|---:|---:|---:|---:|---:|
{chr(10).join(recovery_rows)}

Mean final recovery decreased from {control_final['mean']:.3f} at full retention to {low_final['mean']:.3f} at 25% retention.

| Retention | Paired final difference from full retention |
|---:|---:|
{chr(10).join(paired_rows)}

The recovery reduction increased at each thinning level.

## Structure

Values are means with 95% t confidence intervals across seeds.

| Retention | Matrix density | Exact-frequency alternatives | Nearest access Jaccard | Co-occurrence cosine |
|---:|---:|---:|---:|---:|
{chr(10).join(structure_rows)}

Column thinning reduced matrix density, nearest access similarity and co-occurrence-profile similarity. Integer scaling increased exact-frequency ambiguity. The experiment estimates the joint effect of lower incidence counts under preserved frequency ordering. The simultaneous ambiguity change limits a separate causal interpretation for sparsity.

All reported values were generated from `results/sparsity-sweep/aggregate.json` by `scripts/write_sparsity_report.py`.
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
