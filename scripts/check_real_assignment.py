from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from reviewer_sensitivity import independent_check

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = []
    for language in ['en', 'lus']:
        for condition in ['tokenisation_only', 'stopword_removal', 'stemming']:
            path = ROOT / f'results/corpus/ihop/{language}-{condition}.pkl'
            with path.open('rb') as stream:
                documents, labels, metadata = pickle.load(stream)
            split = metadata['fixed_split']
            indices_by_partition = [split[name][:200] for name in ['auxiliary_indices', 'client_indices']]
            eligible = [column for column in range(len(labels)) if all(0 < sum(column in documents[i] for i in indices) < len(indices) for indices in indices_by_partition)]
            if len(eligible) < 6:
                raise ValueError('Real-data subset has too few supported columns')
            columns = [eligible[i] for i in np.linspace(0, len(eligible)-1, 6, dtype=int)]
            matrices = []
            for name in ['auxiliary_indices', 'client_indices']:
                indices = split[name][:200]
                matrices.append(np.array([[int(column in documents[i]) for column in columns] for i in indices], dtype=int))
            if any(np.any((m.sum(axis=0)==0) | (m.sum(axis=0)==len(m))) for m in matrices):
                raise ValueError('Selected subset has unsupported volume probabilities')
            result = independent_check(*matrices, require_planted=False)
            result.pop('planted_mapping_recovered')
            result.update(language=language, condition=condition, column_indices=columns,
                          auxiliary_incidence=matrices[0].tolist(), client_incidence=matrices[1].tolist(),
                          source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          scope='Independent likelihood and exhaustive assignment costs on disjoint real-data subsets. Ground-truth recovery and full iterative attack equivalence were unvalidated.')
            cases.append(result)
    output = {'cases': cases, 'case_count':len(cases),
              'maximum_cost_error':max(c['maximum_cost_error'] for c in cases),
              'reference_mismatch_resolved':False,
              'interpretation':'Assignment costs and exhaustive minima were checked. The Enron reference mismatch remained unresolved.'}
    (ROOT / 'results/corpus/reviewer-sensitivity/real-assignment-check.json').write_text(json.dumps(output, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
