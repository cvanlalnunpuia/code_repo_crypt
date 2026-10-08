from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "results" / "ihop_enron_baseline" / "aggregate.json"
OUTPUT = ROOT / "docs" / "enron-baseline.md"


def main() -> None:
    result = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    config = result["configuration"]
    records = result["seed_results"]
    metrics = result["aggregate_metrics"]["query_weighted_recovery"]
    checkpoints = config["attack"]["iteration_checkpoints"]

    rows = []
    for checkpoint in checkpoints:
        item = metrics[str(checkpoint)]
        rows.append(
            "| {checkpoint:,} | {mean:.3f} | {sd:.3f} | {low:.3f} to {high:.3f} |".format(
                checkpoint=checkpoint,
                mean=item["mean"],
                sd=item["sample_standard_deviation"],
                low=item["confidence_interval_low"],
                high=item["confidence_interval_high"],
            )
        )

    total_seconds = sum(record["wall_seconds"] for record in records)
    mean_seconds = total_seconds / len(records)
    metrics_equal = all(
        record["query_weighted_recovery"]
        == record["distinct_query_macro_recovery"]
        for record in records
    )

    text = f"""# Enron baseline

The baseline used {len(records)} seeds, {config['dataset']['client_documents_after_split']:,} client documents, {int(config['dataset']['auxiliary_split'].removeprefix('splitn')):,} disjoint auxiliary documents, {config['dataset']['keyword_count']:,} randomly selected keywords and {config['queries']['count']:,} query observations. Each keyword was queried once. IHOP used volume leakage, a free fraction of {config['attack']['free_fraction']:.2f} and no defence.

| Iterations | Mean recovery | Sample SD | 95% t confidence interval |
|---:|---:|---:|---:|
{chr(10).join(rows)}

Query-weighted recovery and distinct-query macro recovery were equal in every seed because each keyword was queried once. Total attack time was {total_seconds:.1f} s. Mean attack time was {mean_seconds:.1f} s per seed. Runtime was measured sequentially on one machine.

The processed dataset SHA-256 digest was `{result['dataset_sha256']}`. The upstream revision was `{result['upstream_revision']}`. Python hash seed {result['python_hash_seed']} was fixed before interpreter startup. Metric equality check was `{str(metrics_equal).lower()}`.

All values in this report were generated from `results/ihop_enron_baseline/aggregate.json` by `scripts/write_baseline_report.py`.
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
