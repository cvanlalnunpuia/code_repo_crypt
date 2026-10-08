from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from analyse_production_recovery import paired, summarise
from analyse_matched_domain_comparison import holm_adjust


class ProductionRecoveryAnalysisTests(unittest.TestCase):
    def test_constant_paired_gain_has_zero_interval_width(self):
        result = paired([0.5] * 10, [0.25] * 10)
        self.assertEqual(result['mean'], 0.25)
        self.assertEqual(result['confidence_interval'], [0.25, 0.25])
        self.assertEqual(result['exact_sign_flip_p'], 2 / 1024)

    def test_pairing_rejects_unequal_samples(self):
        with self.assertRaisesRegex(ValueError, 'equal lengths'):
            paired([0.5, 0.6], [0.25])

    def test_direction_is_first_minus_second(self):
        result = paired([0.25] * 10, [0.5] * 10)
        self.assertEqual(result['differences'], [-0.25] * 10)

    def test_identical_samples_have_unit_p(self):
        result = paired([0.5] * 10, [0.5] * 10)
        self.assertEqual(result['exact_sign_flip_p'], 1)

    def test_holm_family_is_adjusted_together(self):
        items = [{'exact_sign_flip_p': value} for value in [0.01, 0.02, 0.9]]
        holm_adjust(items)
        self.assertEqual([item['holm_adjusted_p'] for item in items], [0.03, 0.04, 0.9])

    def test_summary_count_and_sample_standard_deviation(self):
        result = summarise([0, 1])
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['mean'], 0.5)
        self.assertAlmostEqual(result['sample_standard_deviation'], 2 ** -0.5)


if __name__ == '__main__':
    unittest.main()
