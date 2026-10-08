from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from corpus_stemming import RepeatedSuffixStemmer


def rule(suffix, priority, exceptions=None):
    return {'suffix': suffix, 'replacement': '', 'minimum_stem_letter_count': 2,
            'priority': priority, 'exceptions': exceptions or [], 'basis': 'Synthetic execution test'}


class StemmerExecutionTests(unittest.TestCase):
    def test_repeated_rules_stop_at_unchanged_base(self):
        engine = RepeatedSuffixStemmer([rule('te', 1), rule('ah', 2)], set(), set())
        self.assertEqual(engine.stem('rootahte'), 'root')
        self.assertEqual(engine.stem('root'), 'root')

    def test_guarded_rule_requires_attested_candidate(self):
        engine = RepeatedSuffixStemmer([rule('in', 1)], {'root'}, {'in'})
        self.assertEqual(engine.stem('rootin'), 'root')
        self.assertEqual(engine.stem('unknownin'), 'unknownin')

    def test_intermediate_exception_protects_lexicalised_base(self):
        engine = RepeatedSuffixStemmer([rule('te', 1), rule('tu', 2, ['hotu'])], set(), set())
        self.assertEqual(engine.stem('hotute'), 'hotu')

    def test_original_exception_remains_effective_after_a_strip(self):
        engine = RepeatedSuffixStemmer([rule('te', 1), rule('na', 2, ['rootnate'])], set(), set())
        self.assertEqual(engine.stem('rootnate'), 'rootna')

    def test_nonshortening_rule_is_rejected(self):
        invalid = dict(rule('te', 1), replacement='te')
        with self.assertRaisesRegex(ValueError, 'strictly shorten'):
            RepeatedSuffixStemmer([invalid], set(), set())


if __name__ == '__main__':
    unittest.main()
