import sys
from pathlib import Path
import unittest
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reviewer_sensitivity import independent_check, restrict
from review_uncertainty import build, decisions
from check_iterative_assignment import check_case


class SensitivityTests(unittest.TestCase):
    def test_iterative_update_audit(self):
        root = Path(__file__).resolve().parents[1]
        fixtures = json.loads((root / 'results/corpus/reviewer-sensitivity/real-assignment-check.json').read_text(encoding='utf-8'))
        result = check_case(fixtures['cases'][0], 0)
        self.assertEqual(result['iterations_checked'], 1000)
        self.assertTrue(result['exhaustive_updates'])
        self.assertTrue(result['exact_checkpoint_reconstruction'])
        self.assertLess(result['maximum_cost_error'], 1e-8)

    def test_saved_uncertainty_reconstruction(self):
        root = Path(__file__).resolve().parents[1]
        saved = json.loads((root / 'results/corpus/reviewer-sensitivity/uncertainty.json').read_text(encoding='utf-8'))
        self.assertEqual(build(), saved)
        for contrast in saved['contrasts']:
            self.assertAlmostEqual(contrast['sd'], np.std(contrast['paired_values'], ddof=1))

    def test_structured_decisions_exclude_correspondence(self):
        result = decisions()
        self.assertEqual(set(result), {'source_sha256','mizo_error_types','scope'})
        self.assertNotIn('paragraphs', result)

    def test_real_binary_fixtures(self):
        root = Path(__file__).resolve().parents[1]
        saved = json.loads((root / 'results/corpus/reviewer-sensitivity/real-assignment-check.json').read_text(encoding='utf-8'))
        for case in saved['cases']:
            result = independent_check(np.array(case['auxiliary_incidence']), np.array(case['client_incidence']), require_planted=False)
            self.assertTrue(result['exhaustive_assignment_check'])
            self.assertEqual(result['maximum_cost_error'], case['maximum_cost_error'])

    def test_independent_likelihood_and_assignment(self):
        result = independent_check()
        self.assertLess(result['maximum_cost_error'], 1e-8)
        self.assertTrue(result['exhaustive_assignment_check'])
        self.assertTrue(result['planted_mapping_recovered'])
        self.assertFalse(result['reference_mismatch_resolved'])

    def test_restricted_labels_and_incidence(self):
        rows, labels, metadata = restrict(([[0, 1], [1, 2], []], ['a', 'b', 'c'], {'test': True}), ['c', 'a'], ['x', 'y'])
        self.assertEqual(rows, [[1], [0], []])
        self.assertEqual(labels, ['x', 'y'])
        self.assertEqual(metadata, {'test': True})


if __name__ == '__main__':
    unittest.main()
