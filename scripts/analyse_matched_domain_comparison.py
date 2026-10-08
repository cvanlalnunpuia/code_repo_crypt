from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy import stats

from corpus_records import read_jsonl
from corpus_tokenisation import tokenise


METRICS = ["query_weighted_recovery", "distinct_keyword_macro_recovery"]
MARKERS = {
    "en": {"god", "israel", "jesus", "lord"},
    "lus": {"israel", "lalpa", "pathian"},
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_index(payload: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (run["dataset_id"], run["distribution"], int(run["seed"])): run
        for run in payload["runs"]
    }


def exact_sign_flip_p(differences: list[float]) -> float:
    observed = abs(statistics.mean(differences))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        value = abs(statistics.mean(sign * difference for sign, difference in zip(signs, differences)))
        extreme += value >= observed - 1e-15
        total += 1
    return extreme / total


def paired_summary(flores: list[float], wmt: list[float]) -> dict[str, Any]:
    if len(flores) != len(wmt) or len(flores) < 2:
        raise ValueError("Paired recovery vectors must have equal length of at least two")
    differences = [left - right for left, right in zip(flores, wmt)]
    count = len(differences)
    mean = statistics.mean(differences)
    sample_sd = statistics.stdev(differences)
    margin = stats.t.ppf(0.975, count - 1) * sample_sd / math.sqrt(count)
    return {
        "pair_count": count,
        "flores_mean": statistics.mean(flores),
        "wmt23_mean": statistics.mean(wmt),
        "mean_difference_flores_minus_wmt23": mean,
        "difference_sample_standard_deviation": sample_sd,
        "difference_95_percent_confidence_interval": [mean - margin, mean + margin],
        "paired_standardised_difference": mean / sample_sd if sample_sd else None,
        "exact_sign_flip_p": exact_sign_flip_p(differences),
        "differences_by_seed": differences,
    }


def holm_adjust(results: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(results), key=lambda item: item[1]["exact_sign_flip_p"])
    running = 0.0
    total = len(results)
    adjusted = [0.0] * total
    for rank, (index, result) in enumerate(ordered):
        candidate = min(1.0, result["exact_sign_flip_p"] * (total - rank))
        running = max(running, candidate)
        adjusted[index] = running
    for result, value in zip(results, adjusted):
        result["holm_adjusted_p"] = value


def token_count_summary(selection: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for language in ("en", "lus"):
        values = [int(pair[f"{language}_word_count"]) for pair in selection["pairs"]]
        output[language] = {
            "count": len(values),
            "minimum": min(values),
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "maximum": max(values),
        }
    return output


def marker_summary(
    records: list[dict[str, Any]],
    selection: dict[str, Any],
    token_config: dict[str, Any],
) -> dict[str, Any]:
    by_id = {record["record_id"]: record for record in records}
    output = {}
    for language in ("en", "lus"):
        count = 0
        for pair in selection["pairs"]:
            record = by_id[pair[language]]
            tokens = set(tokenise(record["title"] + "\n" + record["body"], token_config))
            count += bool(tokens & MARKERS[language])
        total = len(selection["pairs"])
        output[language] = {
            "document_count": total,
            "any_marker_document_count": count,
            "any_marker_document_percentage": 100.0 * count / total,
        }
    return output


def build(root: Path) -> dict[str, Any]:
    flores_path = root / "results" / "corpus" / "flores200-fixed-split-experiments.json"
    wmt_path = root / "results" / "corpus" / "wmt23-flores200-matched-experiments.json"
    flores = json.loads(flores_path.read_text(encoding="utf-8"))
    wmt = json.loads(wmt_path.read_text(encoding="utf-8"))
    if flores["config"]["seeds"] != wmt["config"]["seeds"]:
        raise ValueError("Experiment seeds differ")
    if flores["config"]["query_distributions"] != wmt["config"]["query_distributions"]:
        raise ValueError("Query distributions differ")
    if flores["config"]["iteration_checkpoints"] != wmt["config"]["iteration_checkpoints"]:
        raise ValueError("Iteration checkpoints differ")
    flores_runs = run_index(flores)
    wmt_runs = run_index(wmt)
    if set(flores_runs) != set(wmt_runs):
        raise ValueError("Experiment run keys differ")
    final_checkpoint = max(flores["config"]["iteration_checkpoints"])
    checkpoint_index = flores["config"]["iteration_checkpoints"].index(final_checkpoint)
    comparisons = []
    for dataset in sorted({key[0] for key in flores_runs}):
        for distribution in flores["config"]["query_distributions"]:
            keys = [
                (dataset, distribution, int(seed))
                for seed in flores["config"]["seeds"]
            ]
            for metric in METRICS:
                flores_values = [flores_runs[key]["scores"][checkpoint_index][metric] for key in keys]
                wmt_values = [wmt_runs[key]["scores"][checkpoint_index][metric] for key in keys]
                item = paired_summary(flores_values, wmt_values)
                item.update({
                    "language": dataset.split(":", 1)[0],
                    "dataset_id": dataset,
                    "query_distribution": distribution,
                    "metric": metric,
                    "checkpoint": final_checkpoint,
                    "seeds": [int(seed) for seed in flores["config"]["seeds"]],
                })
                comparisons.append(item)
    holm_adjust(comparisons)

    flores_selection_path = root / "results" / "corpus" / "flores200-selection.json"
    wmt_selection_path = root / "results" / "corpus" / "wmt23-flores200-matched-selection.json"
    flores_selection = json.loads(flores_selection_path.read_text(encoding="utf-8"))
    wmt_selection = json.loads(wmt_selection_path.read_text(encoding="utf-8"))
    token_config = json.loads((root / "configs" / "tokenisation_only.json").read_text(encoding="utf-8"))
    flores_records_path = root / "results" / "corpus" / "flores200-records.jsonl"
    wmt_records_path = root / "results" / "corpus" / "wmt23-records.jsonl"
    return {
        "status": "analysed",
        "comparison": "flores200_minus_wmt23_matched_control",
        "pair_count_per_test": len(flores["config"]["seeds"]),
        "checkpoint": final_checkpoint,
        "holm_family_size": len(comparisons),
        "comparisons": comparisons,
        "corpus_structure": {
            "flores200": {
                "paired_document_count": flores_selection["selected_pair_count"],
                "token_counts": token_count_summary(flores_selection),
                "religious_markers": marker_summary(read_jsonl(flores_records_path), flores_selection, token_config),
            },
            "wmt23_matched": {
                "paired_document_count": wmt_selection["selected_pair_count"],
                "token_counts": token_count_summary(wmt_selection),
                "religious_markers": marker_summary(read_jsonl(wmt_records_path), wmt_selection, token_config),
            },
        },
        "inputs": {
            "flores_experiment": {"path": flores_path.relative_to(root).as_posix(), "sha256": file_sha256(flores_path)},
            "wmt23_experiment": {"path": wmt_path.relative_to(root).as_posix(), "sha256": file_sha256(wmt_path)},
            "flores_selection": {"path": flores_selection_path.relative_to(root).as_posix(), "sha256": file_sha256(flores_selection_path)},
            "wmt23_selection": {"path": wmt_selection_path.relative_to(root).as_posix(), "sha256": file_sha256(wmt_selection_path)},
        },
    }


def format_interval(value: list[float]) -> str:
    return f"[{value[0]:.4f}, {value[1]:.4f}]"


def markdown(result: dict[str, Any]) -> str:
    rows = []
    for item in result["comparisons"]:
        metric = "query-weighted" if item["metric"] == "query_weighted_recovery" else "keyword-macro"
        language = {"en": "English", "lus": "Mizo"}[item["language"]]
        distribution = {"uniform_iid": "uniform IID", "zipf_iid": "Zipf IID"}[item["query_distribution"]]
        rows.append(
            f"| {language} | {distribution} | {metric} | "
            f"{item['flores_mean']:.4f} | {item['wmt23_mean']:.4f} | "
            f"{item['mean_difference_flores_minus_wmt23']:.4f} | "
            f"{format_interval(item['difference_95_percent_confidence_interval'])} | "
            f"{item['exact_sign_flip_p']:.4f} | {item['holm_adjusted_p']:.4f} |"
        )
    structure_rows = []
    for corpus, values in result["corpus_structure"].items():
        for language in ("en", "lus"):
            tokens = values["token_counts"][language]
            markers = values["religious_markers"][language]
            language_label = {"en": "English", "lus": "Mizo"}[language]
            structure_rows.append(
                f"| {corpus} | {language_label} | {tokens['median']:.1f} | {tokens['mean']:.1f} | "
                f"{markers['any_marker_document_count']} of {markers['document_count']} "
                f"({markers['any_marker_document_percentage']:.3f}%) |"
            )
    adjusted_below_005 = [item for item in result["comparisons"] if item["holm_adjusted_p"] < 0.05]
    if len(adjusted_below_005) == 1:
        item = adjusted_below_005[0]
        language = {"en": "English", "lus": "Mizo"}[item["language"]]
        metric = "query-weighted recovery" if item["metric"] == "query_weighted_recovery" else "keyword-macro recovery"
        distribution = {"uniform_iid": "uniform IID", "zipf_iid": "Zipf IID"}[item["query_distribution"]]
        adjusted_text = (
            f"One comparison had a Holm-adjusted p value below 0.05. {language} {distribution} {metric} "
            f"was {item['mean_difference_flores_minus_wmt23']:.4f} higher in FLORES-200 "
            f"(Holm p = {item['holm_adjusted_p']:.4f})."
        )
    else:
        adjusted_text = f"{len(adjusted_below_005)} comparisons had Holm-adjusted p values below 0.05."
    return f"""# Matched corpus comparison

FLORES-200 and the WMT23 control each contained 562 paired documents. Each split contained 187 auxiliary pairs and 375 client pairs. Each language used 500 keywords with non-boundary support in both partitions. The experiment used the same ten seeds and independent and identically distributed (IID) query sequences.

## Recovery

Differences are FLORES-200 minus WMT23 at {result['checkpoint']} IHOP iterations. Confidence intervals used the paired seed differences. Exact two-sided sign-flip tests used all 1,024 sign assignments. Holm adjustment covered the eight final-checkpoint comparisons.

| Language | Queries | Metric | FLORES-200 | WMT23 | Difference | 95% CI | Exact p | Holm p |
|---|---|---|---:|---:|---:|---|---:|---:|
{chr(10).join(rows)}

{adjusted_text}

## Corpus structure

| Corpus | Language | Median tokens | Mean tokens | Documents with religious marker |
|---|---|---:|---:|---:|
{chr(10).join(structure_rows)}

Document count, split size, keyword count, seeds, and query generation were controlled. Document length and document construction still differed between corpora. The comparison estimates the association with the complete corpus choice under this protocol. It does not isolate a causal domain effect.

All values were generated by `scripts/analyse_matched_domain_comparison.py`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FLORES-200 with a matched-size WMT23 control.")
    parser.add_argument("--output", type=Path, default=Path("results/corpus/flores200-wmt23-matched-comparison.json"))
    parser.add_argument("--document", type=Path, default=Path("docs/matched-domain-comparison.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    result = build(root)
    output = root / args.output
    document = root / args.document
    output.parent.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    document.write_text(markdown(result), encoding="utf-8")
    print(f"Analysed {len(result['comparisons'])} matched final-checkpoint comparisons")


if __name__ == "__main__":
    main()
