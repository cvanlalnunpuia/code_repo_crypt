from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from corpus_stemming import ClarifiedSuffixStemmer


def rule(suffix, priority, exceptions=None, minimum=2):
    return {'suffix': suffix, 'replacement': '', 'minimum_stem_letter_count': minimum,
            'priority': priority, 'exceptions': exceptions or []}


class ClarifiedStemmerTests(unittest.TestCase):
    def test_corpus_attestation_is_absent(self):
        self.assertEqual(ClarifiedSuffixStemmer([rule('in', 1)]).stem('unknownin'), 'unknown')

    def test_exceptions_use_current_form(self):
        engine = ClarifiedSuffixStemmer([rule('te', 1), rule('na', 2, ['rootnate'])])
        self.assertEqual(engine.stem('rootnate'), 'root')

    def test_blocked_rule_allows_lower_priority(self):
        engine = ClarifiedSuffixStemmer([rule('puite', 1, ['khawpuite']), rule('te', 2), rule('pui', 3, ['khawpui'])])
        self.assertEqual(engine.stem('khawpuite'), 'khawpui')

    def test_combining_mark_does_not_add_a_letter(self):
        engine = ClarifiedSuffixStemmer([rule('in', 1, minimum=3)])
        self.assertEqual(engine.stem('a\u0302xin'), 'âxin')

    def test_nonshortening_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            ClarifiedSuffixStemmer([dict(rule('te', 1), replacement='te')])

    def test_worked_examples(self):
        rules = json.loads((ROOT / 'resources/stemmers/lus-production.reviewed.json').read_text(encoding='utf-8'))['rules']
        engine = ClarifiedSuffixStemmer(rules)
        for token, expected in [('ṭawngṭainaahte', 'ṭawngṭai'), ('khawpuite', 'khawpui'), ('thlahte', 'thlah'), ('rorêltuten', 'rorêl')]:
            with self.subTest(token=token):
                self.assertEqual(engine.stem(token), expected)
