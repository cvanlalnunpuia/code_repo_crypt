from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from analyse_matched_domain_comparison import file_sha256, holm_adjust, paired_summary, run_index


METRICS = ["query_weighted_recovery", "distinct_keyword_macro_recovery"]


def keyword_size(dataset_id: str) -> int:
    match = re.search(r":k([0-9]+)$", dataset_id)
    if not match:
        raise ValueError(f"Dataset lacks a keyword-size suffix: {dataset_id}")
    return int(match.group(1))


def build(root: Path) -> dict[str, Any]:
    flores_path = root / "results" / "corpus" / "flores200-keyword-sensitivity-experiments.json"
    wmt_path = root / "results" / "corpus" / "wmt23-flores200-matched-keyword-sensitivity-experiments.json"
    main_path = root / "results" / "corpus" / "flores200-wmt23-matched-comparison.json"
    flores = json.loads(flores_path.read_text(encoding="utf-8"))
    wmt = json.loads(wmt_path.read_text(encoding="utf-8"))
    main = json.loads(main_path.read_text(encoding="utf-8"))
    for field in ("seeds", "query_distributions", "iteration_checkpoints"):
        if flores["config"][field] != wmt["config"][field]:
            raise ValueError(f"Sensitivity experiment field differs: {field}")
    flores_runs = run_index(flores)
    wmt_runs = run_index(wmt)
    if set(flores_runs) != set(wmt_runs):
        raise ValueError("Sensitivity run keys differ")
    final_checkpoint = max(flores["config"]["iteration_checkpoints"])
    checkpoint_index = flores["config"]["iteration_checkpoints"].index(final_checkpoint)
    comparisons = []
    datasets = sorted({key[0] for key in flores_runs}, key=lambda value: (keyword_size(value), value))
    for dataset in datasets:
        size = keyword_size(dataset)
        language = dataset.split(":", 1)[0]
        for distribution in flores["config"]["query_distributions"]:
            keys = [(dataset, distribution, int(seed)) for seed in flores["config"]["seeds"]]
            for metric in METRICS:
                flores_values = [flores_runs[key]["scores"][checkpoint_index][metric] for key in keys]
                wmt_values = [wmt_runs[key]["scores"][checkpoint_index][metric] for key in keys]
                item = paired_summary(flores_values, wmt_values)
                item.update({
                    "keyword_count": size,
                    "language": language,
                    "dataset_id": dataset,
                    "query_distribution": distribution,
                    "metric": metric,
                    "checkpoint": final_checkpoint,
                    "seeds": [int(seed) for seed in flores["config"]["seeds"]],
                })
                comparisons.append(item)
    holm_adjust(comparisons)
    main_500 = []
    for item in main["comparisons"]:
        enriched = dict(item)
        enriched["keyword_count"] = 500
        enriched["adjustment_family_size"] = main["holm_family_size"]
        main_500.append(enriched)
    for item in comparisons:
        item["adjustment_family_size"] = len(comparisons)
    focus = [
        item
        for item in comparisons + main_500
        if item["language"] == "en"
        and item["query_distribution"] == "zipf_iid"
        and item["metric"] == "query_weighted_recovery"
    ]
    focus.sort(key=lambda item: item["keyword_count"])
    return {
        "status": "analysed",
        "checkpoint": final_checkpoint,
        "sensitivity_holm_family_size": len(comparisons),
        "sensitivity_comparisons": comparisons,
        "main_500_comparisons": main_500,
        "english_zipf_query_weighted_by_size": focus,
        "inputs": {
            "flores_sensitivity": {"path": flores_path.relative_to(root).as_posix(), "sha256": file_sha256(flores_path)},
            "wmt23_sensitivity": {"path": wmt_path.relative_to(root).as_posix(), "sha256": file_sha256(wmt_path)},
            "main_500_comparison": {"path": main_path.relative_to(root).as_posix(), "sha256": file_sha256(main_path)},
        },
    }


def interval(value: list[float]) -> str:
    return f"[{value[0]:.4f}, {value[1]:.4f}]"


def markdown(result: dict[str, Any]) -> str:
    focus_rows = []
    for item in result["english_zipf_query_weighted_by_size"]:
        focus_rows.append(
            f"| {item['keyword_count']} | {item['flores_mean']:.4f} | {item['wmt23_mean']:.4f} | "
            f"{item['mean_difference_flores_minus_wmt23']:.4f} | "
            f"{interval(item['difference_95_percent_confidence_interval'])} | "
            f"{item['exact_sign_flip_p']:.4f} | {item['holm_adjusted_p']:.4f} | "
            f"{item['adjustment_family_size']} |"
        )
    sensitivity_rows = []
    for item in result["sensitivity_comparisons"]:
        language = {"en": "English", "lus": "Mizo"}[item["language"]]
        distribution = {"uniform_iid": "uniform IID", "zipf_iid": "Zipf IID"}[item["query_distribution"]]
        metric = "query-weighted" if item["metric"] == "query_weighted_recovery" else "keyword-macro"
        sensitivity_rows.append(
            f"| {item['keyword_count']} | {language} | {distribution} | {metric} | "
            f"{item['mean_difference_flores_minus_wmt23']:.4f} | "
            f"{interval(item['difference_95_percent_confidence_interval'])} | "
            f"{item['holm_adjusted_p']:.4f} |"
        )
    focus_significant = [item for item in result["english_zipf_query_weighted_by_size"] if item["holm_adjusted_p"] < 0.05]
    sizes = ", ".join(str(item["keyword_count"]) for item in focus_significant)
    focus_text = (
        f"English Zipf query-weighted recovery had Holm-adjusted p below 0.05 at {sizes} keywords."
        if focus_significant
        else "English Zipf query-weighted recovery had no Holm-adjusted p value below 0.05."
    )
    sensitivity_significant = [item for item in result["sensitivity_comparisons"] if item["holm_adjusted_p"] < 0.05]
    language_counts = {
        language: sum(item["language"] == language for item in sensitivity_significant)
        for language in ("en", "lus")
    }
    sensitivity_text = (
        f"Among the 16 sensitivity comparisons, {len(sensitivity_significant)} had Holm-adjusted p below 0.05. "
        f"The counts were {language_counts['en']} for English and {language_counts['lus']} for Mizo."
    )
    return f"""# Keyword-size sensitivity

The matched FLORES-200 and WMT23 comparison was repeated with 100 and 250 keywords. The 562 paired documents, 187/375 split, ten seeds, query sequences, and 1,000 IHOP iterations were retained. Differences are FLORES-200 minus WMT23.

## English Zipf recovery

| Keywords | FLORES-200 | WMT23 | Difference | 95% CI | Exact p | Holm p | Holm family |
|---:|---:|---:|---:|---|---:|---:|---:|
{chr(10).join(focus_rows)}

{focus_text} The 100-keyword and 250-keyword p values were adjusted across 16 sensitivity comparisons. The 500-keyword p value retains its original eight-comparison family.

## Sensitivity comparisons

| Keywords | Language | Queries | Metric | Difference | 95% CI | Holm p |
|---:|---|---|---|---:|---|---:|
{chr(10).join(sensitivity_rows)}

{sensitivity_text}

Document length and document construction still differed between corpora. The sensitivity analysis changes the keyword count only.

All values were generated by `scripts/analyse_keyword_size_sensitivity.py`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse matched keyword-size sensitivity experiments.")
    parser.add_argument("--output", type=Path, default=Path("results/corpus/keyword-size-sensitivity-comparison.json"))
    parser.add_argument("--document", type=Path, default=Path("docs/keyword-size-sensitivity.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    result = build(root)
    output = root / args.output
    document = root / args.document
    output.parent.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    document.write_text(markdown(result), encoding="utf-8")
    print(f"Analysed {len(result['sensitivity_comparisons'])} keyword-size comparisons")


if __name__ == "__main__":
    main()
