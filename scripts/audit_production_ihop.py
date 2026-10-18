from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from independent_ihop import incidence

ROOT = Path(__file__).resolve().parents[1]


def direct_cost(probabilities, counts, document_count, free_keywords, free_tokens, fixed_keywords, fixed_tokens):
    def smoothed(values):
        values = np.array(values, dtype=float, copy=True)
        for row in values:
            positive = row[row>0]
            row[row==0] = positive.min()/100 if len(positive) else 1e-10
        if np.any(values >= 1):
            raise ValueError('Boundary auxiliary probability')
        return values
    p = smoothed(probabilities.diagonal()[free_keywords][None, :])[0]
    c = counts.diagonal()[free_tokens]
    result = -np.log(p)[:, None]*c[None, :] - np.log1p(-p)[:, None]*(document_count-c[None, :])
    if fixed_keywords:
        p = smoothed(probabilities[np.ix_(fixed_keywords, free_keywords)])
        c = counts[np.ix_(fixed_tokens, free_tokens)]
        result -= np.log(p).T @ c + np.log1p(-p).T @ (document_count-c)
    return result


def run_audited(path, dataset_id, config, distribution, seed):
    sys.path.insert(0, str(ROOT/'vendor/ihop-code'))
    import attacks.ihop as upstream
    from run_fixed_split_experiments import run_one
    factory_original = upstream.get_update_coefficients_functions
    assignment_original = upstream.hungarian
    latest = {}
    record = dict(updates_checked=0, maximum_cost_error=0., maximum_relative_cost_error=0.,
                  maximum_independent_objective_gap=0., ambiguous_initial_assignment=False)
    stream_hash = hashlib.sha256()
    def factory(token_trace, token_info, aux, obs, exp_params):
        original_cost, mapping = factory_original(token_trace, token_info, aux, obs, exp_params)
        positions = {value:index for index, value in enumerate(aux['keywords'])}
        docs = [[positions[value] for value in row if value in positions] for row in aux['dataset']]
        matrix = incidence(docs, len(positions))
        probabilities = (matrix.T @ matrix).toarray()/len(docs)
        unique = {}
        for token, accesses in obs['traces']:
            if token not in unique:
                unique[token] = tuple(accesses)
        observation_documents = [[] for _ in range(obs['ndocs'])]
        for token, accesses in enumerate(unique.values()):
            for document in accesses:
                observation_documents[document].append(token)
        observed = incidence(observation_documents, len(unique))
        counts = (observed.T @ observed).toarray()
        def cost(free_keywords, free_tokens, fixed_keywords, fixed_tokens):
            actual = original_cost(free_keywords, free_tokens, fixed_keywords, fixed_tokens)
            expected = direct_cost(probabilities, counts, obs['ndocs'], free_keywords, free_tokens, fixed_keywords, fixed_tokens)
            error = float(np.max(np.abs(actual-expected)))
            relative = error/max(1., float(np.max(np.abs(actual))))
            if relative > 1e-10:
                raise AssertionError('Production update differed from independent likelihood calculation')
            latest.update(expected=expected, error=error)
            record['maximum_cost_error'] = max(record['maximum_cost_error'], error)
            record['maximum_relative_cost_error'] = max(record['maximum_relative_cost_error'], relative)
            return actual
        return cost, mapping
    def assignment(actual):
        rows, columns = assignment_original(actual)
        expected = latest['expected']
        check_rows, check_columns = linear_sum_assignment(expected)
        gap = float(expected[rows, columns].sum()-expected[check_rows, check_columns].sum())
        bound = 2*len(rows)*latest['error']+1e-7
        if abs(gap) > bound:
            raise AssertionError('Production assignment exceeded its independent optimum rounding bound')
        if record['updates_checked'] == 0:
            record['ambiguous_initial_assignment'] = bool(not np.array_equal(columns, check_columns) and abs(gap)<=bound)
        record['updates_checked'] += 1
        record['maximum_independent_objective_gap'] = max(record['maximum_independent_objective_gap'], abs(gap))
        stream_hash.update(actual.tobytes())
        stream_hash.update(np.asarray([rows, columns], dtype=np.int64).tobytes())
        return rows, columns
    try:
        upstream.get_update_coefficients_functions = factory
        upstream.hungarian = assignment
        result = run_one(path, dataset_id, config, distribution, seed)
    finally:
        upstream.get_update_coefficients_functions = factory_original
        upstream.hungarian = assignment_original
    if record['updates_checked'] != config['iterations']+1:
        raise AssertionError('Missing production update audit')
    record['cost_and_assignment_sha256'] = stream_hash.hexdigest()
    record['scope'] = 'Independent likelihood and assignment-objective checks at every production-scale update. The SciPy assignment library was shared. Tied optimal assignments were permitted.'
    result['cost_audit'] = record
    return result
