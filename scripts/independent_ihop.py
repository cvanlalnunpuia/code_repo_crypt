from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.special import xlogy, xlog1py

from export_ihop_corpus import file_sha256


def incidence(documents, keyword_count):
    rows, columns = [], []
    for row, values in enumerate(documents):
        for column in set(values):
            rows.append(row)
            columns.append(column)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, columns)),
                             shape=(len(documents), keyword_count))


def independent_cost(probabilities, observed_counts, document_count,
                     free_keywords, free_tokens, fixed_keywords, fixed_tokens):
    def term(probability, counts):
        p = np.array(probability[free_keywords], dtype=float, copy=True)
        positive = p[p > 0]
        p[p == 0] = positive.min()/100 if positive.size else 1e-10
        if np.any(p >= 1):
            raise ValueError('Boundary auxiliary probability')
        counts = np.asarray(counts)[free_tokens]
        return -xlogy(counts[None, :], p[:, None]) - xlog1py(document_count-counts[None, :], -p[:, None])
    cost = term(probabilities.diagonal(), observed_counts.diagonal())
    for keyword, token in zip(fixed_keywords, fixed_tokens):
        cost += term(probabilities[keyword], observed_counts[token])
    return cost


def recover(auxiliary_matrix, client_matrix, queries, rng, iterations, free_fraction, checkpoints):
    labels = list(dict.fromkeys(queries))
    label_to_token = {value:index for index, value in enumerate(labels)}
    trace = np.array([label_to_token[value] for value in queries], dtype=int)
    probabilities = (auxiliary_matrix.T @ auxiliary_matrix).toarray()/auxiliary_matrix.shape[0]
    observed = (client_matrix[:, labels].T @ client_matrix[:, labels]).toarray()
    keywords = list(range(auxiliary_matrix.shape[1]))
    tokens = list(range(len(labels)))
    mapping = np.full(len(tokens), -1, dtype=int)
    free_count = int(free_fraction*len(tokens))
    if free_count <= 1:
        raise ValueError('Too few free tokens')
    output = []
    for step in range(iterations+1):
        if step == 0:
            free_tokens, fixed_tokens = tokens, []
            free_keywords, fixed_keywords = keywords, []
        else:
            order = rng.permutation(tokens).tolist()
            free_tokens, fixed_tokens = order[:free_count], order[free_count:]
            fixed_keywords = mapping[fixed_tokens].tolist()
            fixed_set = set(fixed_keywords)
            free_keywords = [value for value in keywords if value not in fixed_set]
        cost = independent_cost(probabilities, observed, client_matrix.shape[0],
                                free_keywords, free_tokens, fixed_keywords, fixed_tokens)
        row, col = linear_sum_assignment(cost)
        mapping[np.asarray(free_tokens)[col]] = np.asarray(free_keywords)[row]
        if step in checkpoints:
            output.append(mapping[trace].tolist())
    return output


def run_independent(path: Path, dataset_id, config, distribution, seed):
    with path.open('rb') as stream:
        documents, labels, metadata = pickle.load(stream)
    fixed = metadata['fixed_split']
    auxiliary = incidence([documents[index] for index in fixed['auxiliary_indices']], len(labels))
    client = incidence([documents[index] for index in fixed['client_indices']], len(labels))
    rng = np.random.RandomState(seed)
    if distribution == 'uniform_iid':
        weights = np.ones(len(labels))/len(labels)
    elif distribution == 'zipf_iid':
        weights = 1/np.arange(1, len(labels)+1, dtype=float)
        weights /= weights.sum()
    else:
        raise ValueError('Unsupported query distribution')
    queries = rng.choice(len(labels), len(labels)*config['query_count_per_keyword'], p=weights).tolist()
    rng.permutation(max(queries)+1)
    checkpoints = config['iteration_checkpoints']
    predictions = recover(auxiliary, client, queries, rng, config['iterations'], config['free_fraction'], checkpoints)
    scores = []
    query_array = np.asarray(queries)
    for values in predictions:
        correct = np.asarray(values) == query_array
        counts = np.bincount(query_array, minlength=len(labels))
        matches = np.bincount(query_array, weights=correct, minlength=len(labels))
        scores.append(dict(query_weighted_recovery=float(correct.mean()),
            distinct_keyword_macro_recovery=float((matches[counts>0]/counts[counts>0]).mean()),
            observed_distinct_keyword_count=int(np.count_nonzero(counts))))
    return dict(dataset_id=dataset_id, dataset_sha256=file_sha256(path), distribution=distribution,
                seed=seed, query_count=len(queries), keyword_count=len(labels), checkpoints=checkpoints,
                scores=scores, implementation='Independent volume-only algorithm using a shared SciPy assignment solver')
