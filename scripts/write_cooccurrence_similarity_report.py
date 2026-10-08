from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "results" / "cooccurrence-similarity-sweep" / "aggregate.json"
OUTPUT = ROOT / "docs" / "enron-cooccurrence-similarity-sweep.md"


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
    conditions = ["original"] + [
        f"budget_{budget}" for budget in config["candidate_set_budgets"]
    ]

    recovery_rows = []
    structure_rows = []
    for condition in conditions:
        summary = aggregate["by_condition"][condition]
        label = "Original" if condition == "original" else str(summary["candidate_set_budget"])
        recovery_rows.append(
            "| {label} | {initial} | {middle} | {final} |".format(
                label=label,
                initial=interval(summary["recovery"]["0"]),
                middle=interval(summary["recovery"]["100"]),
                final=interval(summary["recovery"]["1000"]),
            )
        )
        structure_rows.append(
            "| {label} | {client} | {auxiliary} | {frequency} | {density} | {jaccard} |".format(
                label=label,
                client=interval(summary["client_pair_cooccurrence_cosine"]),
                auxiliary=interval(summary["auxiliary_pair_cooccurrence_cosine"]),
                frequency=interval(
                    summary["structure"][
                        "exact_frequency_alternative_proportion"
                    ]
                ),
                density=interval(summary["structure"]["matrix_density"]),
                jaccard=interval(
                    summary["structure"]["median_nearest_access_jaccard"]
                ),
            )
        )

    paired_rows = []
    for condition in conditions[2:]:
        budget = aggregate["by_condition"][condition]["candidate_set_budget"]
        differences = aggregate[
            "paired_differences_from_random_reconstruction"
        ][condition]
        paired_rows.append(
            "| {budget} | {early} | {middle} | {final} |".format(
                budget=budget,
                early=interval(differences["10"]),
                middle=interval(differences["100"]),
                final=interval(differences["1000"]),
            )
        )

    budget_one = aggregate["by_condition"]["budget_1"]
    budget_high = aggregate["by_condition"]["budget_128"]
    final_high = aggregate["paired_differences_from_random_reconstruction"][
        "budget_128"
    ]["1000"]
    text = f"""# Co-occurrence-profile matching produces small final recovery differences on Enron

The experiment used {len(config['seeds'])} paired seeds, {config['keyword_count']} keywords and {config['pair_count']} disjoint keyword pairs per seed. Each recipient retained its original client and auxiliary frequency. Candidate access sets excluded the paired donor's documents. One candidate set served as the reconstruction control. For each larger budget, the set with the highest cosine to the donor profile was retained.

## Recovery

Values are means with 95% t confidence intervals across seeds.

| Candidate-set budget | 0 iterations | 100 iterations | 1,000 iterations |
|---:|---:|---:|---:|
{chr(10).join(recovery_rows)}

The original matrix is shown as a reference. The profile-matching estimates use the budget-1 reconstruction as their paired control.

| Candidate-set budget | Difference at 10 iterations | Difference at 100 iterations | Difference at 1,000 iterations |
|---:|---:|---:|---:|
{chr(10).join(paired_rows)}

Mean final recovery was {budget_one['recovery']['1000']['mean']:.3f} with one candidate set and {budget_high['recovery']['1000']['mean']:.3f} with 128 candidates. The paired final difference for budget 128 was {final_high['mean']:.3f}, with a 95% confidence interval of [{final_high['confidence_interval_low']:.3f}, {final_high['confidence_interval_high']:.3f}]. Budget 16 had a positive difference at 10 iterations. Its interval was above zero. The difference was near zero at 1,000 iterations.

## Structure

Paired-profile cosine increased in every seed as the candidate budget increased. Values are means with 95% t confidence intervals.

| Candidate-set budget | Client pair cosine | Auxiliary pair cosine | Exact-frequency alternatives | Matrix density | Nearest access Jaccard |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(structure_rows)}

Every reconstructed column retained its original frequency. Matrix density and exact-frequency ambiguity remained fixed. Within-pair access Jaccard was zero under every candidate budget. Increasing paired-profile cosine from {budget_one['client_pair_cooccurrence_cosine']['mean']:.3f} to {budget_high['client_pair_cooccurrence_cosine']['mean']:.3f} produced a small final recovery difference. The intervention covered {2 * config['pair_count']} of {config['keyword_count']} keywords.

All reported values were generated from `results/cooccurrence-similarity-sweep/aggregate.json` by `scripts/write_cooccurrence_similarity_report.py`.
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
