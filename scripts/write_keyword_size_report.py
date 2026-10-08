from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "results" / "keyword-size-sweep" / "aggregate.json"
OUTPUT = ROOT / "docs" / "enron-keyword-size-sweep.md"


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
    sizes = config["keyword_counts"]
    checkpoints = config["attack"]["iteration_checkpoints"]
    seed_count = len(config["seeds"])

    recovery_rows = []
    for size in sizes:
        summaries = aggregate["by_keyword_count"][str(size)]["recovery"]
        cells = [interval(summaries[str(checkpoint)]) for checkpoint in checkpoints]
        recovery_rows.append(
            f"| {size} | " + " | ".join(cells) + " |"
        )

    paired_rows = []
    for pair_name, summaries in aggregate[
        "paired_recovery_differences"
    ].items():
        larger, smaller = pair_name.split("_minus_")
        paired_rows.append(
            f"| {larger} minus {smaller} | "
            f"{interval(summaries[str(checkpoints[-1])])} |"
        )

    structure_rows = []
    for size in sizes:
        summaries = aggregate["by_keyword_count"][str(size)]["structure"]
        structure_rows.append(
            "| {size} | {frequency} | {jaccard} | {cosine} |".format(
                size=size,
                frequency=interval(
                    summaries["exact_frequency_alternative_proportion"]
                ),
                jaccard=interval(
                    summaries["median_nearest_access_jaccard"]
                ),
                cosine=interval(summaries["median_cooccurrence_cosine"]),
            )
        )

    final = {
        size: aggregate["by_keyword_count"][str(size)]["recovery"][
            str(checkpoints[-1])
        ]
        for size in sizes
    }
    checkpoint_100 = {
        size: aggregate["by_keyword_count"][str(size)]["recovery"]["100"]
        for size in sizes
    }

    text = f"""# Keyword-universe size and IHOP recovery on Enron

The controlled sweep used {seed_count} paired seeds and nested keyword sets of {sizes[0]}, {sizes[1]} and {sizes[2]} keywords. Each seed retained the same {config['client_documents_after_split']:,} client documents and {config['selected_documents'] - config['client_documents_after_split']:,} auxiliary documents across sizes. Every selected keyword was queried once. Query-weighted recovery therefore equalled distinct-query macro recovery.

## Recovery

Values are means with 95% t confidence intervals across seeds.

| Keywords | 0 iterations | 10 iterations | 100 iterations | 1,000 iterations |
|---:|---:|---:|---:|---:|
{chr(10).join(recovery_rows)}

At 100 iterations, mean recovery increased from {checkpoint_100[sizes[0]]['mean']:.3f} with {sizes[0]} keywords to {checkpoint_100[sizes[1]]['mean']:.3f} with {sizes[1]} keywords and {checkpoint_100[sizes[2]]['mean']:.3f} with {sizes[2]} keywords. At 1,000 iterations, the corresponding means were {final[sizes[0]]['mean']:.3f}, {final[sizes[1]]['mean']:.3f} and {final[sizes[2]]['mean']:.3f}.

The paired final differences were small relative to their uncertainty.

| Paired comparison | Mean difference and 95% CI |
|---|---:|
{chr(10).join(paired_rows)}

## Structure

The exact-frequency alternative proportion measured the share of keywords with at least one frequency match. The median nearest access Jaccard measured the closest access-pattern alternative for each keyword. Values are means with 95% t confidence intervals.

| Keywords | Exact-frequency alternatives | Nearest access Jaccard | Co-occurrence cosine |
|---:|---:|---:|---:|
{chr(10).join(structure_rows)}

Frequency ambiguity increased with the keyword count. Mean final recovery remained above {min(value['mean'] for value in final.values()):.3f} for every size. The result supports the use of keyword-universe size as a controlled factor in the later English-Mizo comparison. Ten paired seeds limit the precision of the estimates.

All reported values were generated from `results/keyword-size-sweep/aggregate.json` by `scripts/write_keyword_size_report.py`.
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
