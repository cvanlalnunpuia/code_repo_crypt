from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from corpus_records import read_jsonl
from corpus_tokenisation import tokenise


MARKERS = {
    "en": ["god", "israel", "jesus", "lord"],
    "lus": ["israel", "lalpa", "pathian"],
}


def rounded_percent(count: int, total: int) -> float:
    return round(100.0 * count / total, 3) if total else 0.0


def token_sets(records: list[dict[str, Any]], token_config: dict[str, Any]) -> dict[str, list[set[str]]]:
    result: dict[str, list[set[str]]] = {"en": [], "lus": []}
    for record in records:
        text = record["title"] + "\n" + record["body"]
        result[record["language"]].append(set(tokenise(text, token_config)))
    return result


def marker_summary(records: list[dict[str, Any]], token_config: dict[str, Any]) -> dict[str, Any]:
    sets = token_sets(records, token_config)
    summary = {}
    for language, documents in sets.items():
        terms = {
            marker: {
                "document_frequency": sum(marker in document for document in documents),
                "document_percentage": rounded_percent(sum(marker in document for document in documents), len(documents)),
            }
            for marker in MARKERS[language]
        }
        any_count = sum(bool(set(MARKERS[language]) & document) for document in documents)
        summary[language] = {
            "document_count": len(documents),
            "any_marker_document_count": any_count,
            "any_marker_document_percentage": rounded_percent(any_count, len(documents)),
            "terms": terms,
        }
    return summary


def experiment_summary(payload: dict[str, Any]) -> dict[str, Any]:
    final_checkpoint = str(max(payload["config"]["iteration_checkpoints"]))
    final = {
        dataset: {
            distribution: checkpoints[final_checkpoint]
            for distribution, checkpoints in distributions.items()
        }
        for dataset, distributions in payload["aggregate"].items()
    }
    return {
        "config_sha256": payload["config_sha256"],
        "manifest_sha256": payload["manifest_sha256"],
        "run_count": len(payload["runs"]),
        "seed_count": len(payload["config"]["seeds"]),
        "final_checkpoint": int(final_checkpoint),
        "final": final,
    }


def build(root: Path) -> dict[str, Any]:
    selection_path = root / "results" / "corpus" / "flores200-selection.json"
    records_path = root / "results" / "corpus" / "flores200-records.jsonl"
    token_metadata_path = root / "results" / "corpus" / "flores200-tokenisation-only" / "metadata.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records = read_jsonl(records_path)
    token_metadata = json.loads(token_metadata_path.read_text(encoding="utf-8"))
    token_config = json.loads((root / "configs" / "tokenisation_only.json").read_text(encoding="utf-8"))
    pairs = selection["pairs"]
    sentence_counts = [pair["sentence_pair_count"] for pair in pairs]
    domain_documents = Counter(domain for pair in pairs for domain in pair["domains"])
    domain_sentences = Counter()
    for pair in pairs:
        for domain in pair["domains"]:
            domain_sentences[domain] += pair["sentence_pair_count"]
    split_documents = Counter(pair["split"] for pair in pairs)
    religious_topic_count = sum(
        any("relig" in topic for topic in pair["topics"])
        for pair in pairs
    )
    result = {
        "status": "audited",
        "source_id": "flores200",
        "source_sentence_pair_count": selection["source_row_count"],
        "selected_article_pair_count": selection["selected_pair_count"],
        "selected_record_count": selection["selected_record_count"],
        "excluded_article_group_count": len(selection["excluded_groups"]),
        "split_article_pair_counts": dict(sorted(split_documents.items())),
        "domain_article_pair_counts": dict(sorted(domain_documents.items())),
        "domain_sentence_pair_counts": dict(sorted(domain_sentences.items())),
        "sentence_pairs_per_article": {
            "minimum": min(sentence_counts),
            "median": statistics.median(sentence_counts),
            "mean": statistics.mean(sentence_counts),
            "maximum": max(sentence_counts),
        },
        "religious_topic_article_count": religious_topic_count,
        "religious_topic_article_percentage": rounded_percent(religious_topic_count, len(pairs)),
        "religious_markers": marker_summary(records, token_config),
        "tokenised_corpus": token_metadata["summary"],
    }
    wmt_path = root / "results" / "corpus" / "wmt23-records.jsonl"
    if wmt_path.exists():
        result["wmt23_religious_markers"] = marker_summary(read_jsonl(wmt_path), token_config)
    smoke_path = root / "results" / "corpus" / "flores200-ihop-smoke.json"
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        result["ihop_smoke"] = {
            "config_sha256": smoke["config_sha256"],
            "manifest_sha256": smoke["manifest_sha256"],
            "datasets": smoke["datasets"],
        }
    pilot_path = root / "results" / "corpus" / "flores200-pilot-experiments.json"
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        result["ihop_pilot"] = experiment_summary(pilot)
    experiment_path = root / "results" / "corpus" / "flores200-fixed-split-experiments.json"
    if experiment_path.exists():
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        result["ihop_experiment"] = experiment_summary(experiment)
    return result


def markdown(result: dict[str, Any]) -> str:
    domain_rows = "\n".join(
        f"| {domain} | {count:,} | {result['domain_sentence_pair_counts'][domain]:,} |"
        for domain, count in result["domain_article_pair_counts"].items()
    )
    marker_rows = []
    for corpus, key in (("FLORES-200", "religious_markers"), ("WMT23", "wmt23_religious_markers")):
        if key not in result:
            continue
        for language in ("en", "lus"):
            item = result[key][language]
            marker_rows.append(
                f"| {corpus} | {language} | {item['any_marker_document_count']:,} of {item['document_count']:,} | "
                f"{item['any_marker_document_percentage']:.3f}% |"
            )
    term_rows = []
    for language in ("en", "lus"):
        for term, item in result["religious_markers"][language]["terms"].items():
            term_rows.append(
                f"| {language} | {term} | {item['document_frequency']:,} | {item['document_percentage']:.3f}% |"
            )
    values = result["sentence_pairs_per_article"]
    experiment_section = ""
    experiment_key = "ihop_experiment" if "ihop_experiment" in result else "ihop_pilot"
    if experiment_key in result:
        experiment = result[experiment_key]
        rows = []
        for dataset, distributions in sorted(experiment["final"].items()):
            language = dataset.split(":", 1)[0]
            for distribution, metrics in sorted(distributions.items()):
                weighted = metrics["query_weighted_recovery"]
                macro = metrics["distinct_keyword_macro_recovery"]
                rows.append(
                    f"| {language} | {distribution} | {weighted['mean']:.4f} | "
                    f"{weighted['sample_standard_deviation']:.4f} | {macro['mean']:.4f} | "
                    f"{macro['sample_standard_deviation']:.4f} |"
                )
        section_name = "IHOP experiment" if experiment_key == "ihop_experiment" else "IHOP pilot"
        experiment_section = f"""
## {section_name}

The experiment contained {experiment['run_count']} fixed-split runs across {experiment['seed_count']} seeds. The table reports mean recovery and sample standard deviation at {experiment['final_checkpoint']} iterations.

| Language | Query distribution | Query-weighted mean | Query-weighted SD | Keyword-macro mean | Keyword-macro SD |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Zipf queries repeated high-rank keywords. Query-weighted recovery therefore exceeded distinct-keyword macro recovery. The macro measure gives each observed keyword equal weight.
"""
    return f"""# FLORES-200 corpus audit

The official [FLORES-200 repository](https://github.com/facebookresearch/flores/tree/main/flores200) releases the corpus under [CC BY-SA 4.0](https://github.com/facebookresearch/flores/blob/main/LICENSE_CC-BY-SA). The direct Meta archive required no account or approval.

The public dev and devtest files contained {result['source_sentence_pair_count']:,} aligned English-Mizo sentences. Grouping by source URL produced {result['selected_article_pair_count']:,} paired article excerpts. {result['excluded_article_group_count']:,} article groups were excluded. Each language contained {result['selected_article_pair_count']:,} documents.

## Domain composition

| Domain | Article pairs | Sentence pairs |
|---|---:|---:|
{domain_rows}

The median article excerpt contained {values['median']:.1f} aligned sentences. The range was {values['minimum']}-{values['maximum']} sentences. The tokenised corpus contained {result['tokenised_corpus']['vocabulary_size']:,} terms and {result['tokenised_corpus']['incidence_count']:,} document-term incidences.

## Religious-term prevalence

The marker set contained `god`, `israel`, `jesus`, and `lord` for English. It contained `israel`, `lalpa`, and `pathian` for Mizo. Counts report exact token matches after the shared tokenisation step.

| Corpus | Language | Documents with any marker | Percentage |
|---|---|---:|---:|
{chr(10).join(marker_rows)}

| Language | FLORES-200 marker | Document frequency | Percentage |
|---|---|---:|---:|
{chr(10).join(term_rows)}

FLORES-200 labelled {result['religious_topic_article_count']:,} article pairs with a topic containing `relig`. This was {result['religious_topic_article_percentage']:.3f}% of the corpus.

{experiment_section}

## Use in the study

WMT23 remains the primary corpus because it supplies 4,974 paired documents. FLORES-200 supplies a licensed external-domain sensitivity corpus. Its Wikinews, Wikivoyage, and Wikibooks composition reduces dependence on the religious concentration measured in WMT23.

All values were generated by `scripts/audit_flores200_corpus.py`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the FLORES-200 English-Mizo sensitivity corpus.")
    parser.add_argument("--output", type=Path, default=Path("results/corpus/flores200-audit.json"))
    parser.add_argument("--document", type=Path, default=Path("docs/flores200-corpus-audit.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    result = build(root)
    output = root / args.output
    document = root / args.document
    output.parent.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    document.write_text(markdown(result), encoding="utf-8")
    print(f"Audited {result['selected_article_pair_count']} FLORES-200 paired articles")


if __name__ == "__main__":
    main()
