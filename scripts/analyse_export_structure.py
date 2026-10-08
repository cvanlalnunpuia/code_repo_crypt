from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse

from compare_preprocessing_structure import gini, quantiles

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = [
    'results/corpus/ihop/manifest.json',
    'results/corpus/flores200-length-balanced-ihop/manifest.json',
    'results/corpus/wmt23-length-balanced-ihop/manifest.json',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(matrix):
    matrix = matrix.astype(np.int32)
    n, k = matrix.shape
    frequency = np.asarray(matrix.sum(axis=0)).ravel()
    intersections = (matrix.T @ matrix).toarray()
    left, right = np.triu_indices(k, 1)
    denominator = frequency[:, None] + frequency[None, :] - intersections
    jaccard = np.divide(intersections, denominator, out=np.zeros((k, k)), where=denominator != 0)
    np.fill_diagonal(jaccard, -1)
    sizes = np.unique(frequency, return_counts=True)[1]
    _, inverse, counts = np.unique(frequency, return_inverse=True, return_counts=True)
    cooccurrence = intersections.astype(float)
    np.fill_diagonal(cooccurrence, 0)
    norms = np.linalg.norm(cooccurrence, axis=1)
    denominator = norms[:, None] * norms[None, :]
    cosine = np.divide(cooccurrence @ cooccurrence.T, denominator,
                       out=np.zeros((k, k)), where=denominator != 0)
    return {'document_count': n, 'keyword_count': k, 'incidence_count': matrix.nnz,
            'density': matrix.nnz / (n * k),
            'frequency_gini': gini(frequency),
            'frequency_cv': float(np.std(frequency) / np.mean(frequency)),
            'frequency_alternative_proportion': float(np.mean(counts[inverse] > 1)),
            'largest_frequency_group': int(max(sizes)),
            'identical_access_pair_count': int(np.count_nonzero(jaccard[left, right] == 1)),
            'nearest_access_jaccard_median': float(np.median(jaccard.max(axis=1))),
            'cooccurrence_cosine_median': float(np.median(cosine[left, right])),
            'document_keyword_count_quantiles': quantiles(np.asarray(matrix.sum(axis=1)).ravel(), [0.25, 0.5, 0.75])}


def build():
    records = []
    for relative in MANIFESTS:
        path = ROOT / relative
        manifest = json.loads(path.read_text(encoding='utf-8'))
        for dataset_id, entry in sorted(manifest['datasets'].items()):
            source = path.parent / entry['path']
            if digest(source) != entry['sha256']:
                raise ValueError(f'Dataset digest changed {source}')
            with source.open('rb') as stream:
                documents, keywords, metadata = pickle.load(stream)
            rows = [row for row, words in enumerate(documents) for _ in words]
            columns = [word for words in documents for word in words]
            matrix = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, columns)),
                                      shape=(len(documents), len(keywords)))
            if not np.all(matrix.data == 1):
                raise ValueError('Export incidence matrix must be binary')
            fixed = metadata['fixed_split']
            auxiliary = matrix[fixed['auxiliary_indices']]
            client = matrix[fixed['client_indices']]
            fa = np.asarray(auxiliary.sum(axis=0)).ravel() / auxiliary.shape[0]
            fc = np.asarray(client.sum(axis=0)).ravel() / client.shape[0]
            ca = (auxiliary.T @ auxiliary).toarray() / auxiliary.shape[0]
            cc = (client.T @ client).toarray() / client.shape[0]
            tri = np.triu_indices(len(keywords), 1)
            records.append({'manifest': relative, 'manifest_sha256': digest(path),
                            'dataset_id': dataset_id, 'dataset_sha256': digest(source),
                            'keyword_sha256': hashlib.sha256('\n'.join(keywords).encode('utf-8')).hexdigest(),
                            'auxiliary': profile(auxiliary), 'client': profile(client),
                            'marginal_probability_rmse': float(np.sqrt(np.mean((fa - fc) ** 2))),
                            'cooccurrence_probability_rmse': float(np.sqrt(np.mean((ca[tri] - cc[tri]) ** 2)))})
    return {'records': records, 'interpretation': 'Descriptive measurements of the exact attack exports. Corpus and split remained fixed across attack seeds.'}


def report(result):
    lines = ['# Structure', '',
             'Table 1 shows client-index measurements from the exact keyword sets supplied to IHOP. Frequency alternatives were keywords sharing a document frequency with another selected keyword. Access similarity was measured between document incidence columns. Co-occurrence similarity was measured between off-diagonal keyword co-occurrence profiles.', '',
             'Table 1. Client-index structure and auxiliary-client marginal probability differences.', '',
             '| Collection | Dataset | Density | Frequency alternatives | Identical access pairs | Median nearest Jaccard | Median cosine | Marginal RMSE |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for record in result['records']:
        p = record['client']
        collection = 'Production' if record['manifest'] == MANIFESTS[0] else ('FLORES balanced' if 'flores200' in record['manifest'] else 'WMT23 balanced')
        lines.append(f"| {collection} | {record['dataset_id']} | {p['density']:.6f} | {p['frequency_alternative_proportion']:.4f} | {p['identical_access_pair_count']} | {p['nearest_access_jaccard_median']:.4f} | {p['cooccurrence_cosine_median']:.4f} | {record['marginal_probability_rmse']:.4f} |")
    lines.extend(['', 'Root mean squared error (RMSE) described auxiliary-client probability differences for the same selected keywords. These measurements were descriptive. Each production index remained unchanged across attack seeds. Repeated attack seeds supplied repeated recovery measurements for the same structural observation.', '',
                  'A multi-corpus predictive model was not fitted. The corpus set and fixed partitions provided limited independent structural observations. The available Enron interventions were analysed as separate paired endpoint contrasts.', '',
                  'All values were recomputed by scripts/analyse_export_structure.py.', ''])
    return '\n'.join(lines)


def main():
    result = build()
    (ROOT / 'results/corpus/export-structure-analysis.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (ROOT / 'docs/export-structure.md').write_text(report(result), encoding='utf-8')
    print(f"Measured {len(result['records'])} exact keyword exports")


if __name__ == '__main__':
    main()
