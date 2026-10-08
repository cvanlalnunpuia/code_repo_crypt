from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_records import deduplicate, load_manifest, normalise_record, normalise_text, normalise_url, read_jsonl, write_jsonl
from article_parsers import PARSERS, parse_article
from source_policy import SourcePolicyError, assert_acquisition_allowed, validate_source_entry
from source_discovery import parse_dipr_listing
from corpus_matching import build_selection, select_matched_pairs, word_count
from corpus_tokenisation import build_from_files, tokenise
from corpus_stopwords import build_stopword_condition, load_stopword_condition, load_reviewed_stopwords
from corpus_stemming import build_stemming_condition, evaluate_annotations, load_stemmer, load_stemming_condition
from compare_preprocessing_structure import build_comparison, build_report
from corpus_split import build_split, build_split_from_files
from export_ihop_corpus import build_export, build_exports
from run_ihop_corpus_smoke import run_smoke, validate_volume_support
from run_fixed_split_experiments import aggregate_runs, generate_queries, recovery_scores
from check_production_readiness import build_report as build_readiness_report, check_readiness
from build_wmt23_corpus import build as build_wmt23_corpus
from build_flores200_corpus import build as build_flores200_corpus
from audit_flores200_corpus import build as audit_flores200_corpus
from build_matched_size_control import build as build_matched_size_control
from analyse_matched_domain_comparison import build as build_matched_domain_comparison
from analyse_keyword_size_sensitivity import build as build_keyword_size_sensitivity
from build_length_balanced_control import build as build_length_balanced_control
from analyse_length_balanced_comparison import build as build_length_balanced_comparison
from prepare_linguistic_review import build as build_linguistic_review
from validate_corpus import validate_record


class CorpusPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = load_manifest(ROOT / "corpus" / "sources" / "source-manifest.json")
        cls.raw = read_jsonl(ROOT / "corpus" / "fixtures" / "raw-articles.jsonl")

    def test_unicode_and_url_normalisation(self) -> None:
        self.assertEqual(normalise_text("A\u0302\r\n\r\n  test"), "Â\n\ntest")
        self.assertEqual(
            normalise_url("HTTPS://Example.COM:443/a//b/?utm_source=x&b=2&a=1#part"),
            "https://example.com/a/b?a=1&b=2",
        )

    def test_normalised_fixture_records_validate(self) -> None:
        records = [normalise_record(item, self.sources[item["source_id"]]) for item in self.raw]
        for record in records:
            self.assertEqual(validate_record(record, False), [])
        self.assertEqual(records[0]["canonical_url"], "https://fixtures.invalid/news/one")

    def test_disabled_source_is_rejected(self) -> None:
        raw = dict(self.raw[0], source_id="vanglaini")
        with self.assertRaises(PermissionError):
            normalise_record(raw, self.sources["vanglaini"])

    def test_acquisition_gate_rejects_disabled_and_unlisted_hosts(self) -> None:
        with self.assertRaises(SourcePolicyError):
            assert_acquisition_allowed(
                self.sources["zalen"],
                "https://www.zalen.in/article/1",
                "lus",
                set(PARSERS),
            )
        with self.assertRaises(SourcePolicyError):
            assert_acquisition_allowed(
                self.sources["fixture_news"],
                "https://example.org/article/1",
                "en",
                set(PARSERS),
            )

    def test_source_manifest_entries_validate(self) -> None:
        for source in self.sources.values():
            self.assertEqual(validate_source_entry(source), [])

    def test_wmt23_archive_is_approved_without_crawler_review(self) -> None:
        source = self.sources["wmt23_indicnecorp"]
        self.assertEqual(source["source_kind"], "archive")
        self.assertEqual(source["permission_status"], "approved")
        self.assertIsNone(source["robots_review"])
        self.assertEqual(validate_source_entry(source), [])

    def test_wmt23_grouping_rebuilds_the_recorded_selection(self) -> None:
        records, selection = build_wmt23_corpus(ROOT / "configs" / "wmt23_indicnecorp.json")
        self.assertEqual(selection["source_row_count"], 50000)
        self.assertEqual(selection["valid_aligned_row_count"], 49748)
        self.assertEqual(selection["selected_pair_count"], 4974)
        self.assertEqual(len(records), 9948)
        self.assertEqual(
            sum(item["reason"] == "empty_aligned_text" for item in selection["excluded_rows"]),
            14,
        )
        self.assertEqual(
            sum(item["reason"] == "empty_after_tokenisation" for item in selection["excluded_rows"]),
            58,
        )
        self.assertEqual(
            sum(item["reason"] == "duplicate_aligned_pair" for item in selection["excluded_rows"]),
            180,
        )

    def test_flores200_archive_is_approved_under_cc_by_sa(self) -> None:
        source = self.sources["flores200"]
        self.assertEqual(source["source_kind"], "archive")
        self.assertEqual(source["permission_status"], "approved")
        self.assertEqual(source["redistribution_mode"], "full_text")
        self.assertIn("LICENSE_CC-BY-SA", source["terms_url"])
        self.assertEqual(validate_source_entry(source), [])

    def test_flores200_url_grouping_rebuilds_the_recorded_selection(self) -> None:
        records, selection = build_flores200_corpus(ROOT / "configs" / "flores200.json")
        self.assertEqual(selection["source_row_count"], 2009)
        self.assertEqual(selection["selected_pair_count"], 562)
        self.assertEqual(selection["selected_record_count"], 1124)
        self.assertEqual(selection["excluded_groups"], [])
        self.assertEqual(len({item["record_id"] for item in records}), 1124)
        self.assertEqual(
            {split: sum(pair["split"] == split for pair in selection["pairs"]) for split in ("dev", "devtest")},
            {"dev": 281, "devtest": 281},
        )

    def test_wmt23_matched_control_uses_flores_pair_count(self) -> None:
        control = build_matched_size_control(ROOT / "configs" / "wmt23_flores200_matched_control.json")
        self.assertEqual(control["input_pair_count"], 4974)
        self.assertEqual(control["target_pair_count"], 562)
        self.assertEqual(control["selected_pair_count"], 562)
        self.assertEqual(control["selected_record_count"], 1124)
        for rank, pair in enumerate(control["pairs"], 1):
            self.assertEqual(pair["matched_control_rank"], rank)

    def test_matched_domain_comparison_uses_paired_final_results(self) -> None:
        result = build_matched_domain_comparison(ROOT)
        self.assertEqual(result["checkpoint"], 1000)
        self.assertEqual(result["pair_count_per_test"], 10)
        self.assertEqual(result["holm_family_size"], 8)
        self.assertEqual(len(result["comparisons"]), 8)
        self.assertTrue(all(len(item["differences_by_seed"]) == 10 for item in result["comparisons"]))
        adjusted = [item for item in result["comparisons"] if item["holm_adjusted_p"] < 0.05]
        self.assertEqual(len(adjusted), 1)
        self.assertEqual(adjusted[0]["language"], "en")
        self.assertEqual(adjusted[0]["query_distribution"], "zipf_iid")
        self.assertEqual(adjusted[0]["metric"], "query_weighted_recovery")

    def test_keyword_size_sensitivity_covers_two_additional_sizes(self) -> None:
        result = build_keyword_size_sensitivity(ROOT)
        self.assertEqual(result["checkpoint"], 1000)
        self.assertEqual(result["sensitivity_holm_family_size"], 16)
        self.assertEqual(len(result["sensitivity_comparisons"]), 16)
        self.assertEqual(
            [item["keyword_count"] for item in result["english_zipf_query_weighted_by_size"]],
            [100, 250, 500],
        )

    def test_length_balanced_control_links_bilingual_pairs(self) -> None:
        result = build_length_balanced_control(ROOT / "configs" / "length_balanced_control.json")
        self.assertEqual(result["matching"]["matched_pair_count"], 253)
        self.assertEqual(len(result["split_indices"]["auxiliary"]), 84)
        self.assertEqual(len(result["split_indices"]["client"]), 169)
        self.assertLessEqual(result["matching"]["ratio_summary"]["en"]["maximum"], 1.25)
        self.assertLessEqual(result["matching"]["ratio_summary"]["lus"]["maximum"], 1.25)
        self.assertEqual(
            [pair["length_match_id"] for pair in result["selections"]["flores"]["pairs"]],
            [pair["length_match_id"] for pair in result["selections"]["wmt23"]["pairs"]],
        )

    def test_length_balanced_comparison_covers_three_keyword_sizes(self) -> None:
        result = build_length_balanced_comparison(ROOT)
        self.assertEqual(result["matched_pair_count"], 253)
        self.assertEqual(result["holm_family_size"], 24)
        self.assertEqual(len(result["comparisons"]), 24)
        self.assertEqual(
            [item["keyword_count"] for item in result["english_zipf_query_weighted_comparison"]],
            [100, 250, 500],
        )

    def test_linguistic_review_records_assigned_reviewer(self) -> None:
        document = build_linguistic_review(ROOT, ROOT / "configs" / "linguistic_review.json")
        self.assertIn("Reviewer: C Lalrinawma", document)
        self.assertEqual(document.count("0 pending token decisions"), 2)

    def test_flores200_audit_records_external_domain_difference(self) -> None:
        audit = audit_flores200_corpus(ROOT)
        self.assertEqual(audit["domain_article_pair_counts"], {"wikibooks": 179, "wikinews": 207, "wikivoyage": 176})
        self.assertEqual(audit["religious_markers"]["en"]["any_marker_document_count"], 10)
        self.assertEqual(audit["religious_markers"]["lus"]["any_marker_document_count"], 10)
        self.assertEqual(audit["ihop_pilot"]["run_count"], 40)
        self.assertEqual(audit["ihop_pilot"]["final_checkpoint"], 100)
        self.assertEqual(audit["ihop_experiment"]["run_count"], 40)
        self.assertEqual(audit["ihop_experiment"]["final_checkpoint"], 1000)

    def test_fixture_html_parser_excludes_boilerplate(self) -> None:
        html = (ROOT / "corpus" / "fixtures" / "article-en.html").read_text(encoding="utf-8")
        record = parse_article("fixture_html_v1", html)
        self.assertEqual(record["title"], "Council approves river survey")
        self.assertNotIn("Navigation", record["body"])
        self.assertNotIn("Related", record["body"])

    def test_dipr_parser_extracts_mizo_article_fields(self) -> None:
        html = (ROOT / "corpus" / "fixtures" / "dipr-mizo-article.html").read_text(encoding="utf-8")
        record = parse_article("dipr_html_v1", html)
        self.assertEqual(record["title"], "Tuihna enfiahna kalpui a ni")
        self.assertEqual(record["authors"], ["Entirna Ziaktu, MIS"])
        self.assertEqual(record["section"], "Mizo Press Releases")
        self.assertEqual(record["published_at"], "2026-01-03T08:00:00+05:30")
        self.assertNotIn("Latest press release", record["body"])
        self.assertNotIn("Government site footer", record["body"])

    def test_dipr_has_project_authorisation_and_collection_stays_gated(self) -> None:
        source = self.sources["dipr_mizoram"]
        self.assertEqual(source["project_authorisation"]["status"], "approved")
        self.assertEqual(source["parser_id"], "dipr_html_v1")
        with self.assertRaises(SourcePolicyError):
            assert_acquisition_allowed(
                source,
                "https://dipr.mizoram.gov.in/post/example",
                "lus",
                set(PARSERS),
            )

    def test_dipr_listing_parser_returns_sorted_unique_article_urls(self) -> None:
        html = (ROOT / "corpus" / "fixtures" / "dipr-mizo-listing.html").read_text(encoding="utf-8")
        self.assertEqual(
            parse_dipr_listing(html),
            [
                "https://dipr.mizoram.gov.in/post/khawtlang-hmalakna",
                "https://dipr.mizoram.gov.in/post/tuihna-enfiahna",
            ],
        )

    def test_fixture_pilot_selects_one_unique_bilingual_pair(self) -> None:
        result = build_selection(
            ROOT / "corpus" / "fixtures" / "deduplicated-articles.jsonl",
            ROOT / "configs" / "fixture_corpus_pilot.json",
        )
        self.assertEqual(result["selected_pair_count"], 1)
        self.assertEqual(result["selected_record_count"], 2)
        self.assertLessEqual(result["pairs"][0]["date_gap_days"], 7)
        self.assertLessEqual(result["pairs"][0]["length_ratio"], 2)
        self.assertEqual(len({result["pairs"][0]["lus"], result["pairs"][0]["en"]}), 2)

    def test_matching_is_independent_of_input_order(self) -> None:
        records = read_jsonl(ROOT / "corpus" / "fixtures" / "deduplicated-articles.jsonl")
        config = json.loads((ROOT / "configs" / "fixture_corpus_pilot.json").read_text(encoding="utf-8"))
        self.assertEqual(select_matched_pairs(records, config), select_matched_pairs(list(reversed(records)), config))

    def test_word_count_retains_mizo_tokens(self) -> None:
        self.assertEqual(word_count({"body": "Mizo ṭawng thumal pathum"}), 4)

    def test_shared_tokeniser_preserves_diacritics_and_apostrophes(self) -> None:
        config = json.loads((ROOT / "configs" / "tokenisation_only.json").read_text(encoding="utf-8"))
        self.assertEqual(
            tokenise("A\u0302n MIZO a'n don't state-of-the-art 2026 x", config),
            ["ân", "mizo", "a'n", "don't", "state", "of", "the", "art"],
        )

    def test_tokenised_fixture_matrix_matches_selected_documents(self) -> None:
        result, metadata = build_from_files(
            ROOT / "corpus" / "fixtures" / "deduplicated-articles.jsonl",
            ROOT / "corpus" / "fixtures" / "fixture-pilot-selection.json",
            ROOT / "configs" / "tokenisation_only.json",
        )
        self.assertEqual(result["summary"]["document_count"], 2)
        self.assertEqual(result["summary"]["language_document_counts"], {"en": 1, "lus": 1})
        self.assertEqual(result["matrix"].shape[0], 2)
        self.assertEqual(result["matrix"].shape[1], result["summary"]["vocabulary_size"])
        self.assertEqual(result["matrix"].nnz, result["summary"]["incidence_count"])
        self.assertEqual(metadata["pipeline"], "tokenisation_only")

    def test_stopword_condition_removes_only_approved_entries(self) -> None:
        result, metadata = build_stopword_condition(
            ROOT / "corpus" / "fixtures" / "deduplicated-articles.jsonl",
            ROOT / "corpus" / "fixtures" / "fixture-pilot-selection.json",
            ROOT / "configs" / "stopword_removal_fixture.json",
        )
        self.assertEqual(sum(item["removed_stopword_count"] for item in result["documents"]), 14)
        vocabulary = {item["token"] for item in result["vocabulary"]}
        self.assertNotIn("the", vocabulary)
        self.assertNotIn("chuan", vocabulary)
        self.assertIn("survey", vocabulary)
        self.assertIn("tuihna", vocabulary)
        self.assertEqual(metadata["pipeline"], "stopword_removal")

    def test_pending_production_lists_cannot_run(self) -> None:
        token_config = json.loads((ROOT / 'configs/tokenisation_only.json').read_text(encoding='utf-8'))
        with self.assertRaisesRegex(ValueError, "pending"):
            load_reviewed_stopwords(ROOT / 'resources/stopwords/lus-production.pending.json', 'lus', 'production', token_config)

    def test_reviewed_production_lists_preserve_returned_decisions(self) -> None:
        _, _, stopwords, reviews = load_stopword_condition(ROOT / 'configs/stopword_removal_production.json')
        self.assertEqual(len(stopwords['en']), 116)
        self.assertEqual(len(stopwords['lus']), 117)
        self.assertEqual(reviews['lus']['reviewed_by'], 'C Lalrinawma')
        self.assertEqual(reviews['lus']['reviewed_at'], '2026-10-06')

    def test_stopword_candidates_require_review(self) -> None:
        for language in ("en", "lus"):
            payload = json.loads((ROOT / "corpus" / "fixtures" / f"{language}-stopword-candidates.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "pending")
            self.assertTrue(payload["entries"])
            self.assertTrue(all(entry["decision"] == "pending" for entry in payload["entries"]))

    def test_fixture_stemmers_apply_rules_and_exceptions(self) -> None:
        english, _ = load_stemmer(ROOT / "resources" / "stemmers" / "en-fixture.json", "en", "synthetic_fixture")
        mizo, _ = load_stemmer(ROOT / "resources" / "stemmers" / "lus-fixture.json", "lus", "synthetic_fixture")
        self.assertEqual(english("approved"), "approv")
        self.assertEqual(mizo("enfiahna"), "enfiah")
        self.assertEqual(mizo("hmunte"), "hmun")
        self.assertEqual(mizo("mana"), "mana")
        self.assertEqual(mizo("tute"), "tute")

    def test_stemming_condition_builds_matrix(self) -> None:
        result, metadata = build_stemming_condition(
            ROOT / "corpus" / "fixtures" / "deduplicated-articles.jsonl",
            ROOT / "corpus" / "fixtures" / "fixture-pilot-selection.json",
            ROOT / "configs" / "stemming_fixture.json",
        )
        self.assertEqual(sum(item["changed_stem_count"] for item in result["documents"]), 9)
        self.assertEqual(result["summary"]["vocabulary_size"], 40)
        self.assertEqual(metadata["pipeline"], "stopword_removal_with_stemming")

    def test_fixture_stemmer_annotations_are_recomputed(self) -> None:
        result = evaluate_annotations(
            ROOT / "corpus" / "annotations" / "stemmer-fixture.json",
            ROOT / "configs" / "stemming_fixture.json",
        )
        self.assertEqual(result["overall"], {"item_count": 8, "correct_count": 8, "accuracy": 1.0})

    def test_pending_production_stemming_condition_cannot_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "pending"):
            load_stemmer(ROOT / 'resources/stemmers/lus-production.pending.json', 'lus', 'production')

    def test_preprocessing_structural_comparison_covers_each_condition_and_language(self) -> None:
        result = build_comparison(ROOT / "configs" / "preprocessing_structural_comparison.json")
        self.assertEqual(set(result["profiles"]), {"tokenisation_only", "stopword_removal", "stemming"})
        for condition in result["profiles"].values():
            self.assertEqual(set(condition), {"en", "lus"})
            for profile in condition.values():
                self.assertEqual(profile["document_count"], 1)
                self.assertGreater(profile["selected_keyword_count"], 1)
                self.assertEqual(profile["matrix_density"], 1.0)

    def test_preprocessing_report_is_generated_from_comparison(self) -> None:
        result = build_comparison(ROOT / "configs" / "preprocessing_structural_comparison.json")
        report = build_report(result)
        self.assertIn("| Tokenisation only | lus | 1 | 32 |", report)
        self.assertIn("| Stopwords and stemming | en | 1 | 18 |", report)

    def test_fixture_split_is_shared_and_reproducible(self) -> None:
        result = build_split_from_files(ROOT / "configs" / "corpus_split_fixture.json")
        self.assertEqual(result["mode"], "shared_fixture_only")
        for assignment in result["assignments"].values():
            self.assertEqual(assignment["auxiliary_record_ids"], assignment["client_record_ids"])

    def test_disjoint_split_keeps_language_pairs_together(self) -> None:
        pairs = [
            {"lus": f"lus-{index}", "en": f"en-{index}"}
            for index in range(5)
        ]
        selection = {"config": {"languages": ["lus", "en"]}, "pairs": pairs}
        config = {
            "config_version": "1.0.0",
            "mode": "disjoint_by_matched_pair",
            "auxiliary_fraction": 0.4,
            "seed": 7,
        }
        result = build_split(selection, config)
        self.assertEqual(len(result["auxiliary_pair_indices"]), 2)
        self.assertFalse(set(result["auxiliary_pair_indices"]) & set(result["client_pair_indices"]))
        for language in ("lus", "en"):
            assignment = result["assignments"][language]
            self.assertFalse(set(assignment["auxiliary_record_ids"]) & set(assignment["client_record_ids"]))

    def test_ihop_export_matches_upstream_tuple_shape(self) -> None:
        split = json.loads((ROOT / "corpus" / "fixtures" / "fixture-split.json").read_text(encoding="utf-8"))
        dataset, keywords, auxiliary = build_export(
            ROOT / "corpus" / "fixtures" / "tokenisation-only",
            "tokenisation_only",
            "lus",
            split,
            500,
        )
        self.assertEqual(len(dataset), 1)
        self.assertEqual(len(keywords), 32)
        self.assertEqual(auxiliary["fixed_split"]["auxiliary_indices"], [0])
        self.assertEqual(auxiliary["fixed_split"]["client_indices"], [0])

    def test_volume_smoke_rejects_boundary_access_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "boundary access probability"):
            validate_volume_support([[0, 1]], 2, "fixture")

    def test_flores200_export_has_volume_support_in_both_splits(self) -> None:
        split = json.loads((ROOT / "results" / "corpus" / "flores200-split.json").read_text(encoding="utf-8"))
        for language in ("lus", "en"):
            dataset, keywords, auxiliary = build_export(
                ROOT / "results" / "corpus" / "flores200-tokenisation-only",
                "tokenisation_only",
                language,
                split,
                500,
                True,
            )
            fixed = auxiliary["fixed_split"]
            auxiliary_documents = [dataset[index] for index in fixed["auxiliary_indices"]]
            client_documents = [dataset[index] for index in fixed["client_indices"]]
            self.assertEqual(len(keywords), 500)
            validate_volume_support(auxiliary_documents, 500, f"{language} auxiliary")
            validate_volume_support(client_documents, 500, f"{language} client")

    def test_multi_size_export_records_each_keyword_universe(self) -> None:
        _, manifest = build_exports(ROOT / "configs" / "ihop_corpus_export_flores200_keyword_sensitivity.json")
        self.assertEqual(
            set(manifest["datasets"]),
            {
                "en:tokenisation_only:k100",
                "en:tokenisation_only:k250",
                "lus:tokenisation_only:k100",
                "lus:tokenisation_only:k250",
            },
        )
        self.assertEqual(
            sorted(item["keyword_count"] for item in manifest["datasets"].values()),
            [100, 100, 250, 250],
        )

    def test_frequency_smoke_executes_all_exports(self) -> None:
        result = run_smoke(ROOT / "configs" / "ihop_corpus_smoke.json")
        self.assertEqual(result["status"], "executed")
        self.assertEqual(len(result["datasets"]), 6)
        self.assertTrue(all(len(item["recovery"]) == 2 for item in result["datasets"].values()))

    def test_query_generation_and_recovery_metrics(self) -> None:
        np.random.seed(3)
        each = generate_queries("each", 5, 2)
        self.assertEqual(sorted(each), [0, 1, 2, 3, 4])
        np.random.seed(3)
        uniform = generate_queries("uniform_iid", 5, 2)
        self.assertEqual(len(uniform), 10)
        scores = recovery_scores([0, 1, 0, 2], [0, 1, 2, 2])
        self.assertEqual(scores["query_weighted_recovery"], 0.75)
        self.assertEqual(scores["distinct_keyword_macro_recovery"], 5 / 6)

    def test_fixed_split_fixture_experiment_count(self) -> None:
        result = json.loads((ROOT / "corpus" / "fixtures" / "fixed-split-experiments.json").read_text(encoding="utf-8"))
        self.assertEqual(len(result["runs"]), 36)
        self.assertEqual(len(result["aggregate"]), 6)
        self.assertTrue(all(set(value) == {"each", "uniform_iid", "zipf_iid"} for value in result["aggregate"].values()))

    def test_production_readiness_lists_all_current_gates(self) -> None:
        result = check_readiness(ROOT / "configs" / "production_readiness.json")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["blocker_count"], 0)
        codes = {item["code"] for item in result["blockers"]}
        self.assertNotIn("source_permission_missing", codes)
        self.assertNotIn("robots_review_missing", codes)
        self.assertNotIn("stopwords_pending_lus", codes)
        self.assertNotIn("stemmers_pending_lus", codes)
        self.assertNotIn("artifact_missing", codes)
        self.assertIn("Unmet gates: 0", build_readiness_report(result))

    def test_exact_and_near_duplicates_are_marked(self) -> None:
        records = [normalise_record(item, self.sources[item["source_id"]]) for item in self.raw]
        output = deduplicate(records, 0.75)
        counts = {status: sum(item["duplicate"]["status"] == status for item in output) for status in ("unique", "exact", "near")}
        self.assertEqual(counts, {"unique": 2, "exact": 1, "near": 1})
        for record in output:
            self.assertEqual(validate_record(record, True), [])

    def test_jsonl_output_is_deterministic(self) -> None:
        records = [normalise_record(item, self.sources[item["source_id"]]) for item in self.raw]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            write_jsonl(first, records)
            write_jsonl(second, records)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            for line in first.read_text(encoding="utf-8").splitlines():
                json.loads(line)


if __name__ == "__main__":
    unittest.main()
