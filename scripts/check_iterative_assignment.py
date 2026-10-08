from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]


def independent_cost(auxiliary, client, queries, free_kw, free_tags, fixed_kw, fixed_tags):
    probabilities = auxiliary.T @ auxiliary / len(auxiliary)
    observed = client[:, queries].T @ client[:, queries]
    n = len(client)
    costs = np.zeros((len(free_kw), len(free_tags)))
    terms = [(np.diag(probabilities), np.diag(observed))]
    terms.extend((probabilities[kw], observed[tag]) for kw, tag in zip(fixed_kw, fixed_tags))
    for pvec, counts in terms:
        p = np.asarray(pvec[free_kw], dtype=float).copy()
        p[p == 0] = min(p[p > 0])/100 if np.any(p > 0) else 1e-10
        c = counts[free_tags]
        log_choose = binom.logpmf(c, n, .5) + n*np.log(2)
        costs -= binom.logpmf(c[None, :], n, p[:, None]) - log_choose[None, :]
    return costs


def exhaustive(cost):
    scored = [(float(sum(cost[perm[c],c] for c in range(cost.shape[1]))), perm)
              for perm in itertools.permutations(range(cost.shape[0]), cost.shape[1])]
    minimum = min(x[0] for x in scored)
    options = [perm for value, perm in scored if np.isclose(value, minimum, atol=1e-8, rtol=0)]
    return minimum, options


def check_case(case, seed, iterations=1000):
    sys.path.insert(0, str(ROOT / 'vendor/ihop-code'))
    import attacks.ihop as implementation
    from defense import generate_observations
    from exp_params import ExpParams
    auxiliary = np.array(case['auxiliary_incidence'], dtype=int)
    client = np.array(case['client_incidence'], dtype=int)
    queries = [3,0,5,1,4,2]
    docs = lambda m: [np.flatnonzero(row).tolist() for row in m]
    aux = {'dataset':docs(auxiliary),'keywords':list(range(6)), 'frequencies':np.ones(6)/6,'mode_query':'each'}
    params = ExpParams()
    params.set_defense_params('none')
    params.set_attack_params('ihop', mode='Vol', niters=iterations, pfree=.5)
    checkpoints = [0,10,100,iterations]
    params.att_params['niter_list'] = checkpoints
    obs, _, _ = generate_observations({'dataset':docs(client),'keywords':list(range(6)),'frequencies':None}, params.def_params, queries)
    records = []
    original_factory, original_assignment = implementation.get_update_coefficients_functions, implementation.hungarian
    def factory(*args):
        compute, labels = original_factory(*args)
        def wrapped(free_kw, free_tags, fixed_kw, fixed_tags):
            actual_cost = compute(free_kw, free_tags, fixed_kw, fixed_tags)
            records.append(dict(free_kw=list(free_kw), free_tags=list(free_tags), fixed_kw=list(fixed_kw), fixed_tags=list(fixed_tags), actual_cost=actual_cost.copy()))
            return actual_cost
        return wrapped, labels
    def assignment(cost):
        rows, cols = original_assignment(cost)
        records[-1]['rows'], records[-1]['cols'] = rows.tolist(), cols.tolist()
        return rows, cols
    try:
        implementation.get_update_coefficients_functions, implementation.hungarian = factory, assignment
        np.random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            actual = implementation.ihop_attack(obs, aux, params)
    finally:
        implementation.get_update_coefficients_functions, implementation.hungarian = original_factory, original_assignment
    ties, maximum_error = 0, 0.
    mapping, expected = [None]*6, []
    for step, record in enumerate(records):
        cost = independent_cost(auxiliary, client, queries, record['free_kw'], record['free_tags'], record['fixed_kw'], record['fixed_tags'])
        maximum_error = max(maximum_error, float(np.max(np.abs(cost-record['actual_cost']))))
        minimum, choices = exhaustive(cost)
        selected = float(cost[record['rows'],record['cols']].sum())
        if not np.isclose(selected, minimum, atol=1e-8, rtol=0):
            raise AssertionError('Iterative update failed independent exhaustive optimum')
        ties += int(len(choices)>1)
        for row, col in zip(record['rows'], record['cols']):
            mapping[record['free_tags'][col]] = record['free_kw'][row]
        if step in checkpoints:
            expected.append(mapping.copy())
    if maximum_error > 1e-8 or expected != actual or len(records) != iterations+1:
        raise AssertionError('Iterative cost or checkpoint reconstruction differed')
    return dict(language=case['language'], condition=case['condition'], seed=seed,
                checkpoints=checkpoints, iterations_checked=len(records)-1, tie_updates=ties,
                maximum_cost_error=maximum_error, exact_checkpoint_reconstruction=True,
                exhaustive_updates=True, free_fraction=.5)


def main():
    source = ROOT / 'results/corpus/reviewer-sensitivity/real-assignment-check.json'
    fixtures = json.loads(source.read_text(encoding='utf-8'))
    cases = [check_case(case, seed) for case in fixtures['cases'] for seed in [0,1,2]]
    result = dict(cases=cases, case_count=len(cases), fixture_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  iterations_per_case=1000, seed_count=3,
                  scope='All recorded small-fixture update costs and selected assignment minima were checked independently. Checkpoints were reconstructed from the recorded updates. Tied optimal assignments were accepted. An independently selected trajectory and production-scale equivalence were unvalidated. The Enron reference mismatch remained unresolved.')
    (ROOT / 'results/corpus/reviewer-sensitivity/iterative-check.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({'case_count':len(cases),'passed':True,'tie_updates':sum(x['tie_updates'] for x in cases)}))


if __name__ == '__main__':
    main()
