import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyse_export_structure import profile
from audit_wmt23_corpus import summary


class ExportStructureTests(unittest.TestCase):
    def test_identical_columns_are_counted_as_one_pair(self):
        matrix = sparse.csr_matrix(np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]]))
        result = profile(matrix)
        self.assertEqual(result['identical_access_pair_count'], 1)
        self.assertEqual(result['frequency_alternative_proportion'], 1)
        self.assertAlmostEqual(result['density'], 6 / 9)

    def test_fractional_ratio_extrema_are_preserved(self):
        result = summary([1.125, 1.875])
        self.assertEqual(result['minimum'], 1.125)
        self.assertEqual(result['maximum'], 1.875)


if __name__ == '__main__':
    unittest.main()
