from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "structure" / "enron-full.json"
OUTPUT = ROOT / "docs" / "enron-structural-profile.md"


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    schema = result["schema"]
    frequency = result["document_frequency"]
    ambiguity = result["frequency_ambiguity"]
    access = result["access_pattern_similarity"]
    cooccurrence = result["cooccurrence_profile_similarity"]

    ambiguity_rows = []
    for tolerance in result["configuration"][
        "frequency_ambiguity_tolerances_documents"
    ]:
        item = ambiguity[str(tolerance)]
        ambiguity_rows.append(
            "| {tolerance:,} | {pairs:,} | {pair_pct:.2%} | {keywords:,} | {keyword_pct:.2%} |".format(
                tolerance=tolerance,
                pairs=item["ambiguous_pair_count"],
                pair_pct=item["ambiguous_pair_proportion"],
                keywords=item["keywords_with_alternative_count"],
                keyword_pct=item["keywords_with_alternative_proportion"],
            )
        )

    access_rows = []
    for threshold in result["configuration"]["access_similarity_thresholds"]:
        proportion = access["threshold_pair_proportions"][f"{threshold:g}"]
        access_rows.append(f"| {threshold:.2f} | {proportion:.6%} |")

    text = f"""# Enron structural profile

The processed Enron index contains {schema['document_count']:,} documents, {schema['keyword_count']:,} keywords and {schema['incidence_count']:,} non-zero document-keyword entries. Matrix density is {schema['matrix_density']:.4%}. Documents contain a mean of {result['document_length']['mean']:.2f} indexed keywords and a median of {result['document_length']['quantiles']['0.5']:.1f}.

## Frequency structure

Mean document frequency is {frequency['mean']:.2f}. Median document frequency is {frequency['quantiles']['0.5']:.1f}. The Gini coefficient is {frequency['gini_coefficient']:.3f}. The population coefficient of variation is {frequency['population_coefficient_of_variation']:.3f}. The most frequent 1% of keywords account for {frequency['top_1_percent_incidence_share']:.2%} of incidences. The most frequent 10% account for {frequency['top_10_percent_incidence_share']:.2%}.

| Frequency tolerance in documents | Ambiguous pairs | Pair proportion | Keywords with an alternative | Keyword proportion |
|---:|---:|---:|---:|---:|
{chr(10).join(ambiguity_rows)}

There are {result['exact_frequency_groups']['non_singleton_group_count']:,} non-singleton exact-frequency groups. The largest contains {result['exact_frequency_groups']['largest_group_size']:,} keywords.

## Access patterns

Median pairwise Jaccard similarity is {access['all_pair_quantiles']['0.5']:.4f}. Median nearest-neighbour similarity is {access['nearest_neighbour_quantiles']['0.5']:.4f}. No two keyword columns have identical access patterns.

| Jaccard threshold | Pair proportion at or above threshold |
|---:|---:|
{chr(10).join(access_rows)}

## Co-occurrence profiles

The deterministic sample contains {cooccurrence['sample_size']:,} distinct keyword pairs. Mean cosine similarity is {cooccurrence['mean']:.3f}. Median similarity is {cooccurrence['quantiles']['0.5']:.3f}. The sample standard deviation is {cooccurrence['sample_standard_deviation']:.3f}. The sample seed is {cooccurrence['sample_seed']}.

All values in this report were generated from `results/structure/enron-full.json` by `scripts/write_structure_report.py`. Metric definitions are stored in `docs/structural-metrics.md`.
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
