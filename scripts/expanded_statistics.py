from __future__ import annotations

import statistics

import numpy as np

from analyse_matched_domain_comparison import holm_adjust
from analyse_production_recovery import summarise


def exact_sign_flip_chunked(differences):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not 2 <= len(values) <= 24 or not np.all(np.isfinite(values)):
        raise ValueError('Expected two to twenty-four finite paired differences')
    observed = abs(statistics.mean(differences))
    extreme = 0
    total = 2**len(values)
    for start in range(0, total, 32768):
        bits = (np.arange(start, min(start+32768, total), dtype=np.uint32)[:, None] >> np.arange(len(values), dtype=np.uint32)) & 1
        signs = bits.astype(np.int8)*2-1
        scores = np.abs(signs @ values)/len(values)
        extreme += int(np.count_nonzero(scores >= observed-1e-15))
    return extreme/total


def original_extension_statistics(records, config):
    seeds = config['original_partition_attack_seeds']
    lookup = {(r['task']['language'], r['task']['condition'], r['distribution'], r['seed']):r['scores'][-1]
              for r in records if r['task']['family']=='original_extended'}
    metrics = ['query_weighted_recovery', 'distinct_keyword_macro_recovery']
    distributions = ['uniform_iid', 'zipf_iid']
    conditions = ['tokenisation_only', 'stopword_removal', 'stemming']
    def values(language, condition, distribution, metric):
        return [lookup[(language, condition, distribution, seed)][metric] for seed in seeds]
    def comparison(left, right):
        differences = [a-b for a,b in zip(left, right)]
        result = summarise(differences)
        result.update(differences=differences, exact_sign_flip_p=exact_sign_flip_chunked(differences))
        return result
    language = []
    for condition in conditions:
        for distribution in distributions:
            for metric in metrics:
                row = comparison(values('lus', condition, distribution, metric), values('en', condition, distribution, metric))
                row.update(condition=condition, distribution=distribution, metric=metric)
                language.append(row)
    preprocessing = []
    for lang in ['en', 'lus']:
        for left, right in [('stopword_removal','tokenisation_only'), ('stemming','tokenisation_only'), ('stemming','stopword_removal')]:
            for distribution in distributions:
                for metric in metrics:
                    row = comparison(values(lang, left, distribution, metric), values(lang, right, distribution, metric))
                    row.update(language=lang, left=left, right=right, distribution=distribution, metric=metric)
                    preprocessing.append(row)
    holm_adjust(language)
    holm_adjust(preprocessing)
    return dict(seeds=seeds, language_comparisons=language, preprocessing_comparisons=preprocessing,
                minimum_two_sided_probability=2/2**len(seeds),
                scope='Exploratory extensions of the original partition. The original confirmatory results and comparison families were preserved separately.')
