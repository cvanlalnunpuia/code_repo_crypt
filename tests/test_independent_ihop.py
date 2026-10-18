from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from independent_ihop import incidence, independent_cost, recover
from check_iterative_assignment import independent_cost as binomial_cost
from audit_production_ihop import direct_cost
from auxiliary_controls import select_terms
from expanded_statistics import exact_sign_flip_chunked
from analyse_matched_domain_comparison import exact_sign_flip_p


class IndependentIHOPTests(unittest.TestCase):
    def test_incidence_is_binary(self):
        self.assertEqual(incidence([[0, 0, 2], [1]], 3).toarray().tolist(), [[1, 0, 1], [0, 1, 0]])

    def test_cost_matches_independent_binomial_fixture(self):
        rng = np.random.RandomState(812)
        auxiliary = rng.binomial(1, .3, (200, 6))
        client = rng.binomial(1, .4, (210, 6))
        queries = [3, 0, 5, 1, 4, 2]
        p = auxiliary.T @ auxiliary/len(auxiliary)
        observed = client[:, queries].T @ client[:, queries]
        args = ([1, 2, 4, 5], [0, 1, 2, 3], [0, 3], [4, 5])
        calculated = independent_cost(p, observed, len(client), *args)
        expected = binomial_cost(auxiliary, client, queries, *args)
        np.testing.assert_allclose(calculated, expected, atol=1e-8, rtol=0)
        np.testing.assert_allclose(direct_cost(p, observed, len(client), *args), expected, atol=1e-8, rtol=0)

    def test_auxiliary_keyword_selection(self):
        self.assertEqual(select_terms([{'a':5, 'b':8, 'c':1}, {'a':5, 'b':1, 'd':10}], 2), ['a', 'b'])
        with self.assertRaises(ValueError):
            select_terms([{'a':1}, {'b':1}], 1)

    def test_chunked_sign_flip_matches_original(self):
        for values in [[0., 0.], [.1, -.1, .2], [.1]*10, [.1, -.2, .3, 0., -.5]]:
            self.assertEqual(exact_sign_flip_chunked(values), exact_sign_flip_p(values))
        self.assertEqual(exact_sign_flip_chunked([.1]*20), 2/2**20)

    def test_planted_recovery_and_repeatability(self):
        rng = np.random.RandomState(417)
        data = sparse.csr_matrix(rng.binomial(1, np.linspace(.15, .7, 6), (200, 6)))
        queries = [3, 0, 5, 1, 4, 2]
        a = recover(data, data, queries, np.random.RandomState(0), 30, .5, [0, 30])
        b = recover(data, data, queries, np.random.RandomState(0), 30, .5, [0, 30])
        self.assertEqual(a, b)
        self.assertEqual(a[-1], queries)


if __name__ == '__main__':
    unittest.main()
