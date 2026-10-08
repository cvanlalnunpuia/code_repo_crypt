from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Callable

from nltk.stem.snowball import SnowballStemmer

from corpus_records import read_jsonl
from corpus_stopwords import load_stopword_condition, resolve_project_path
from corpus_tokenisation import build_tokenised_corpus, file_sha256, write_tokenised_corpus


STEMMING_PIPELINE_VERSION = "2.0.0"


def letter_count(value: str) -> int:
    return sum(unicodedata.category(character)[0] in {"L", "M"} for character in value)


class SuffixStemmer:
    def __init__(self, rules: list[dict[str, Any]]) -> None:
        self.rules = sorted(rules, key=lambda rule: (int(rule["priority"]), -len(rule["suffix"]), rule["suffix"]))

    def stem(self, token: str) -> str:
        for rule in self.rules:
            if token in rule["exceptions"] or not token.endswith(rule["suffix"]):
                continue
            base = token[:-len(rule["suffix"])] if rule["suffix"] else token
            candidate = base + rule["replacement"]
            if letter_count(candidate) >= int(rule["minimum_stem_letter_count"]):
                return candidate
        return token


class RepeatedSuffixStemmer(SuffixStemmer):
    """Apply reviewed suffix rules with corpus attestation and rule-scoped exceptions."""

    def __init__(self, rules: list[dict[str, Any]], vocabulary: set[str], guarded_suffixes: set[str]) -> None:
        super().__init__(rules)
        self.vocabulary = vocabulary
        self.guarded_suffixes = guarded_suffixes
        if any(not rule['suffix'] or len(rule['replacement']) >= len(rule['suffix']) for rule in self.rules):
            raise ValueError('Repeated suffix rules must strictly shorten tokens')

    def stem(self, token: str) -> str:
        original = token
        while True:
            changed = False
            for rule in self.rules:
                if original in rule['exceptions'] or token in rule['exceptions'] or not token.endswith(rule['suffix']):
                    continue
                candidate = token[:-len(rule['suffix'])] + rule['replacement']
                if letter_count(candidate) < rule['minimum_stem_letter_count']:
                    continue
                if rule['suffix'] in self.guarded_suffixes and candidate not in self.vocabulary:
                    continue
                token = candidate
                changed = True
                break
            if not changed:
                return token


class ClarifiedSuffixStemmer(SuffixStemmer):
    """Apply the clarified repeated rules with current-form exceptions."""

    def __init__(self, rules: list[dict[str, Any]]) -> None:
        super().__init__(rules)
        if any(not rule['suffix'] or len(rule['replacement']) >= len(rule['suffix']) for rule in self.rules):
            raise ValueError('Repeated suffix rules must strictly shorten tokens')

    def stem(self, token: str) -> str:
        token = unicodedata.normalize('NFC', token)
        while True:
            for rule in self.rules:
                if token in rule['exceptions'] or not token.endswith(rule['suffix']):
                    continue
                candidate = token[:-len(rule['suffix'])] + rule['replacement']
                letters = sum(unicodedata.category(character).startswith('L') for character in candidate)
                if letters < rule['minimum_stem_letter_count']:
                    continue
                token = candidate
                break
            else:
                return token


def validate_stemmer_specification(payload: dict[str, Any], language: str, required_scope: str) -> None:
    if payload.get("language") != language:
        raise ValueError(f"Stemmer language mismatch for {language}")
    if payload.get("scope") != required_scope:
        raise ValueError(f"Stemmer scope mismatch for {language}")
    if payload.get("status") != "approved":
        raise ValueError(f"Stemmer specification is pending for {language}")
    if not payload.get("reviewed_by") or not payload.get("reviewed_at"):
        raise ValueError(f"Stemmer review evidence is incomplete for {language}")
    algorithm = payload.get("algorithm")
    if language == "en" and algorithm != "snowball_english":
        raise ValueError("English stemmer must use snowball_english")
    if language == "lus" and algorithm != "suffix_strip_v1":
        raise ValueError("Mizo stemmer must use suffix_strip_v1")
    seen = set()
    for rule in payload.get("rules", []):
        suffix = unicodedata.normalize("NFC", rule["suffix"]).casefold()
        key = (suffix, int(rule["priority"]))
        if key in seen:
            raise ValueError(f"Duplicate suffix rule {suffix!r}")
        seen.add(key)
        if suffix != rule["suffix"]:
            raise ValueError(f"Suffix lacks NFC case-folded form {rule['suffix']!r}")
        if int(rule["minimum_stem_letter_count"]) < 1:
            raise ValueError("minimum_stem_letter_count must be positive")


def load_stemmer(path: Path, language: str, required_scope: str) -> tuple[Callable[[str], str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_stemmer_specification(payload, language, required_scope)
    if payload["algorithm"] == "snowball_english":
        implementation = SnowballStemmer("english")
        return implementation.stem, payload
    implementation = SuffixStemmer(payload["rules"])
    return implementation.stem, payload


def load_stemming_condition(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, set[str]], dict[str, Callable[[str], str]], dict[str, Any]]:
    condition = json.loads(config_path.read_text(encoding="utf-8"))
    if condition.get("config_version") != "1.0.0":
        raise ValueError("Unsupported stemming config_version")
    stopword_path = resolve_project_path(config_path, condition["stopword_condition"])
    _, tokenisation, stopwords, stopword_review = load_stopword_condition(stopword_path)
    stemmers = {}
    stemmer_review = {}
    for language in ("en", "lus"):
        specification_path = resolve_project_path(config_path, condition["stemmer_specifications"][language])
        stemmers[language], payload = load_stemmer(specification_path, language, condition["required_scope"])
        stemmer_review[language] = {
            "path": condition["stemmer_specifications"][language],
            "sha256": file_sha256(specification_path),
            "specification_version": payload["specification_version"],
            "algorithm": payload["algorithm"],
            "rule_count": len(payload["rules"]),
            "reviewed_by": payload["reviewed_by"],
            "reviewed_at": payload["reviewed_at"],
        }
    if condition.get('mizo_execution_policy'):
        policy_path = resolve_project_path(config_path, condition['mizo_execution_policy'])
        policy = json.loads(policy_path.read_text(encoding='utf-8'))
        if policy.get('policy_version') == '2.0.0' and policy.get('algorithm') == 'reviewer_clarified_repeated_suffix':
            if policy.get('exception_scope') != 'current_form_per_rule' or policy.get('guarded_suffixes') != []:
                raise ValueError('Clarified policy requires current-form exceptions and unguarded suffixes')
            evidence_path = resolve_project_path(config_path, policy['clarification_record'])
            evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
            if evidence['source_sha256'] != policy['clarification_source_sha256']:
                raise ValueError('Clarification evidence digest differs')
            mizo_path = resolve_project_path(config_path, condition['stemmer_specifications']['lus'])
            rules = json.loads(mizo_path.read_text(encoding='utf-8'))['rules']
            stemmers['lus'] = ClarifiedSuffixStemmer(rules).stem
            stemmer_review['lus']['execution_policy'] = {
                'path': condition['mizo_execution_policy'], 'sha256': file_sha256(policy_path),
                'algorithm': policy['algorithm'], 'clarification_sha256': file_sha256(evidence_path),
                'clarification_source_sha256': evidence['source_sha256'],
            }
            review = {"stopwords": stopword_review, "stemmers": stemmer_review}
            return condition, tokenisation, stopwords, stemmers, review
        if policy.get('policy_version') != '1.0.0' or policy.get('algorithm') != 'repeated_suffix_with_attestation':
            raise ValueError('Unsupported Mizo execution policy')
        if policy.get('exception_scope') != 'original_and_intermediate_per_rule':
            raise ValueError('Unsupported exception scope')
        lexicon_path = resolve_project_path(config_path, policy['wordlist'])
        lexicon = json.loads(lexicon_path.read_text(encoding='utf-8'))
        if lexicon.get('partition') != 'auxiliary' or lexicon.get('language') != 'lus':
            raise ValueError('Mizo word list must use auxiliary Mizo documents')
        if file_sha256(lexicon_path) != policy['wordlist_sha256']:
            raise ValueError('Mizo word list digest differs')
        mizo_path = resolve_project_path(config_path, condition['stemmer_specifications']['lus'])
        rules = json.loads(mizo_path.read_text(encoding='utf-8'))['rules']
        if not set(policy['guarded_suffixes']) <= {rule['suffix'] for rule in rules}:
            raise ValueError('Guarded suffixes are absent from the reviewed rules')
        stemmers['lus'] = RepeatedSuffixStemmer(rules, set(lexicon['tokens']), set(policy['guarded_suffixes'])).stem
        stemmer_review['lus']['execution_policy'] = {
            'path': condition['mizo_execution_policy'], 'sha256': file_sha256(policy_path),
            'wordlist_sha256': file_sha256(lexicon_path), 'algorithm': policy['algorithm'],
            'wordlist_partition': lexicon['partition'],
        }
    review = {"stopwords": stopword_review, "stemmers": stemmer_review}
    return condition, tokenisation, stopwords, stemmers, review


def build_stemming_condition(input_path: Path, selection_path: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    condition, tokenisation, stopwords, stemmers, review = load_stemming_condition(config_path)
    records = read_jsonl(input_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result = build_tokenised_corpus(records, selection, tokenisation, stopwords, stemmers)
    metadata = {
        "pipeline": "stopword_removal_with_stemming",
        "pipeline_version": STEMMING_PIPELINE_VERSION,
        "input_sha256": file_sha256(input_path),
        "selection_sha256": file_sha256(selection_path),
        "condition_config_sha256": file_sha256(config_path),
        "condition_config": condition,
        "tokenisation_config": tokenisation,
        "reviews": review,
    }
    return result, metadata


def write_stemming_condition(output_dir: Path, result: dict[str, Any], metadata: dict[str, Any]) -> None:
    write_tokenised_corpus(output_dir, result, metadata)


def evaluate_annotations(annotation_path: Path, config_path: Path) -> dict[str, Any]:
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotations.get("status") != "approved" or not annotations.get("annotator") or not annotations.get("annotated_at"):
        raise ValueError("Stemmer annotations lack approved review evidence")
    condition, _, _, stemmers, _ = load_stemming_condition(config_path)
    if annotations.get("scope") != condition["required_scope"]:
        raise ValueError("Annotation scope differs from the stemming condition")
    rows = []
    totals = {language: {"item_count": 0, "correct_count": 0} for language in ("en", "lus")}
    for item in annotations["items"]:
        predicted = stemmers[item["language"]](item["token"])
        correct = predicted == item["expected_stem"]
        totals[item["language"]]["item_count"] += 1
        totals[item["language"]]["correct_count"] += int(correct)
        rows.append({**item, "predicted_stem": predicted, "correct": correct})
    for values in totals.values():
        values["accuracy"] = values["correct_count"] / values["item_count"] if values["item_count"] else None
    total_items = sum(values["item_count"] for values in totals.values())
    total_correct = sum(values["correct_count"] for values in totals.values())
    return {
        "annotation_sha256": file_sha256(annotation_path),
        "config_sha256": file_sha256(config_path),
        "by_language": totals,
        "overall": {
            "item_count": total_items,
            "correct_count": total_correct,
            "accuracy": total_correct / total_items if total_items else None,
        },
        "items": rows,
    }
