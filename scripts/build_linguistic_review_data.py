from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nltk
import sklearn
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from corpus_records import read_jsonl
from corpus_tokenisation import tokenise


def compact(value: str, limit: int = 180) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def selected_records(root: Path) -> list[dict[str, Any]]:
    selection = json.loads((root / "results/corpus/wmt23-selection.json").read_text(encoding="utf-8"))
    selected = {
        pair[language]
        for pair in selection["pairs"]
        for language in selection["config"]["languages"]
    }
    return [
        record
        for record in read_jsonl(root / "results/corpus/wmt23-records.jsonl")
        if record["record_id"] in selected
    ]


def corpus_counts(records: list[dict[str, Any]], token_config: dict[str, Any]) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]]]:
    document_frequency = {"en": Counter(), "lus": Counter()}
    token_frequency = {"en": Counter(), "lus": Counter()}
    for record in records:
        language = record["language"]
        tokens = tokenise(record["title"] + "\n" + record["body"], token_config)
        document_frequency[language].update(set(tokens))
        token_frequency[language].update(tokens)
    return document_frequency, token_frequency


def suffix_candidates(document_frequency: Counter[str], token_frequency: Counter[str]) -> list[dict[str, Any]]:
    by_suffix: dict[str, list[str]] = defaultdict(list)
    for token in document_frequency:
        if len(token) < 6 or not all(character.isalpha() for character in token):
            continue
        for length in range(2, 6):
            if len(token) - length >= 3:
                by_suffix[token[-length:]].append(token)
    selected = []
    for length in range(2, 6):
        ranked = []
        for suffix, tokens in by_suffix.items():
            if len(suffix) != length or len(tokens) < 8:
                continue
            ranked.append((len(tokens), sum(token_frequency[token] for token in tokens), suffix, tokens))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        for type_count, occurrence_count, suffix, tokens in ranked[:10]:
            examples = sorted(tokens, key=lambda token: (-token_frequency[token], token))[:6]
            selected.append({
                "suffix": suffix,
                "distinct_type_count": type_count,
                "token_occurrence_count": occurrence_count,
                "examples": examples,
                "generation_basis": f"Top corpus ending of length {length} by distinct token types",
            })
    return selected


def sample_tokens(
    language: str,
    document_frequency: Counter[str],
    excluded: set[str],
    suffixes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = [
        (token, count)
        for token, count in document_frequency.most_common()
        if token not in excluded and len(token) >= 4 and all(character.isalpha() for character in token)
    ]
    chosen: list[tuple[str, int, str]] = []
    seen = set()

    def add(token: str, count: int, reason: str) -> None:
        if token not in seen:
            seen.add(token)
            chosen.append((token, count, reason))

    for token, count in ranked[:15]:
        add(token, count, "frequent corpus form")
    middle_start = max(0, len(ranked) // 2 - 8)
    for token, count in ranked[middle_start : middle_start + 15]:
        add(token, count, "middle-frequency corpus form")
    for token, count in reversed(ranked):
        if 2 <= count <= 10:
            add(token, count, "infrequent corpus form")
        if len(chosen) >= 40:
            break
    if language == "lus":
        for item in suffixes:
            for token in item["examples"][:2]:
                add(token, document_frequency[token], f"candidate ending {item['suffix']}")
                if len(chosen) >= 50:
                    break
            if len(chosen) >= 50:
                break
    return [
        {"token": token, "document_frequency": count, "sampling_reason": reason}
        for token, count, reason in chosen[:50 if language == "lus" else 40]
    ]


def concordances(
    records: list[dict[str, Any]],
    token_config: dict[str, Any],
    targets: dict[str, set[str]],
) -> dict[str, dict[str, list[str]]]:
    output = {language: defaultdict(list) for language in targets}
    for record in records:
        language = record["language"]
        for line in (record["title"] + "\n" + record["body"]).splitlines():
            line = compact(line)
            if not line:
                continue
            present = set(tokenise(line, token_config)) & targets[language]
            for token in present:
                if len(output[language][token]) < 2 and line not in output[language][token]:
                    output[language][token].append(line)
    return {language: dict(values) for language, values in output.items()}


def build(root: Path) -> dict[str, Any]:
    token_config = json.loads((root / "configs/tokenisation_only.json").read_text(encoding="utf-8"))
    assignment = json.loads((root / "configs/linguistic_review.json").read_text(encoding="utf-8"))
    records = selected_records(root)
    document_frequency, token_frequency = corpus_counts(records, token_config)
    stopwords = {
        language: json.loads((root / f"resources/stopwords/{language}-production.pending.json").read_text(encoding="utf-8"))
        for language in ("en", "lus")
    }
    suffixes = suffix_candidates(document_frequency["lus"], token_frequency["lus"])
    stopword_tokens = {
        language: {item["token"] for item in payload["entries"]}
        for language, payload in stopwords.items()
    }
    annotations = {
        language: sample_tokens(language, document_frequency[language], stopword_tokens[language], suffixes)
        for language in ("en", "lus")
    }
    targets = {
        language: stopword_tokens[language] | {item["token"] for item in annotations[language]}
        for language in ("en", "lus")
    }
    contexts = concordances(records, token_config, targets)
    for language, payload in stopwords.items():
        for rank, item in enumerate(payload["entries"], 1):
            item["rank"] = rank
            item["document_frequency"] = int(re.search(r"document frequency ([0-9]+) of", item["basis"]).group(1))
            item["contexts"] = contexts[language].get(item["token"], [])
            if language == "en" and item["token"] in ENGLISH_STOP_WORDS:
                item["review_hint"] = "Listed in scikit-learn ENGLISH_STOP_WORDS. Check context before exclusion."
            elif language == "en":
                item["review_hint"] = "Absent from scikit-learn ENGLISH_STOP_WORDS. Content or ambiguity review required."
            else:
                item["review_hint"] = "Native-speaker decision required. Frequency alone does not justify exclusion."
    snowball = SnowballStemmer("english")
    for language, rows in annotations.items():
        for item in rows:
            item["contexts"] = contexts[language].get(item["token"], [])
            if language == "en":
                item["suggested_stem"] = snowball.stem(item["token"])
    return {
        "reviewer": assignment["reviewer"],
        "assigned_at": assignment["assigned_at"],
        "document_count_by_language": dict(Counter(record["language"] for record in records)),
        "stopwords": stopwords,
        "mizo_suffix_candidates": suffixes,
        "annotations": annotations,
        "implementation_versions": {
            "nltk": nltk.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "source_files": [
            "results/corpus/wmt23-records.jsonl",
            "results/corpus/wmt23-selection.json",
            "resources/stopwords/en-production.pending.json",
            "resources/stopwords/lus-production.pending.json",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence for the linguistic review workbook.")
    parser.add_argument("--output", type=Path, default=Path("results/corpus/linguistic-review-data.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    result = build(root)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Prepared {len(result['stopwords']['en']['entries'])} English and "
        f"{len(result['stopwords']['lus']['entries'])} Mizo stopword rows, "
        f"{len(result['mizo_suffix_candidates'])} suffix candidates, and "
        f"{sum(len(rows) for rows in result['annotations'].values())} annotation rows"
    )


if __name__ == "__main__":
    main()
