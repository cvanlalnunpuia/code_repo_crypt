from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import pickle
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import binom

from corpus_split import build_split
from corpus_stemming import load_stemming_condition
from export_ihop_corpus import build_export, file_sha256, write_pickle
from run_fixed_split_experiments import run_one, generate_queries

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/corpus/reviewer-sensitivity'
CONDITIONS = ['tokenisation_only', 'stopword_removal', 'stemming']


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def restrict(payload, terms, labels=None):
    dataset, keywords, metadata = payload
    positions = {keywords.index(term): index for index, term in enumerate(terms)}
    return [[positions[k] for k in row if k in positions] for row in dataset], labels or terms, metadata


def independent_check(binary=None, client_binary=None, require_planted=True):
    sys.path.insert(0, str(ROOT / 'vendor/ihop-code'))
    import utils
    from attacks.ihop import get_update_coefficients_functions
    from defense import generate_observations
    from exp_params import ExpParams
    from processing.process_obs import process_traces
    rng = np.random.RandomState(417)
    if binary is None:
        binary = rng.binomial(1, np.linspace(.15, .7, 6), size=(200, 6))
    if client_binary is None:
        client_binary = binary
    docs = [np.flatnonzero(row).tolist() for row in binary]
    client_docs = [np.flatnonzero(row).tolist() for row in client_binary]
    aux = {'dataset': docs, 'keywords': list(range(6)), 'frequencies': np.ones(6)/6, 'mode_query': 'each'}
    params = ExpParams()
    params.set_defense_params('none')
    params.set_attack_params('ihop', mode='Vol', niters=10, pfree=.5)
    queries = [3, 0, 5, 1, 4, 2]
    obs, _, _ = generate_observations({'dataset': client_docs, 'keywords': list(range(6)), 'frequencies': None}, params.def_params, queries)
    trace, info = process_traces(obs, aux, params.def_params)
    cost, _ = get_update_coefficients_functions(trace, info, aux, obs, params)
    probabilities = binary.T @ binary / len(binary)
    access = [set(info[tag]) for tag in range(6)]
    observed = np.array([[len(a & b) for b in access] for a in access])
    max_error = 0.
    for fixed in [[], [0], [0, 1], [0, 1, 2]]:
        fixed_kw = [queries[t] for t in fixed]
        free_tags = [t for t in range(6) if t not in fixed]
        free_kw = [k for k in range(6) if k not in fixed_kw]
        expected = np.zeros((len(free_kw), len(free_tags)))
        for row, kw in enumerate(free_kw):
            for col, tag in enumerate(free_tags):
                for pvec, counts in [(np.diag(probabilities), np.diag(observed))] + [(probabilities[fk], observed[ft]) for fk, ft in zip(fixed_kw, fixed)]:
                    p = float(pvec[kw])
                    if p == 0:
                        free_probabilities = pvec[free_kw]
                        p = float(min(free_probabilities[free_probabilities > 0]) / 100) if np.any(free_probabilities > 0) else 1e-10
                    n = int(counts[tag])
                    log_choose = float(binom.logpmf(n, len(client_binary), .5) + len(client_binary)*np.log(2))
                    expected[row, col] -= float(binom.logpmf(n, len(client_binary), p) - log_choose)
        actual = cost(free_kw, free_tags, fixed_kw, fixed)
        max_error = max(max_error, float(np.max(np.abs(actual-expected))))
        rows, cols = linear_sum_assignment(actual)
        minimum = min(sum(expected[perm[col], col] for col in range(len(free_tags))) for perm in itertools.permutations(range(len(free_kw))))
        if not np.isclose(actual[rows, cols].sum(), minimum, atol=1e-8, rtol=0):
            raise AssertionError('Assignment differs from exhaustive minimum')
    if max_error > 1e-8:
        raise AssertionError('Independent likelihood check failed')
    rows, cols = linear_sum_assignment(cost(list(range(6)), list(range(6)), [], []))
    recovered = {int(col): int(row) for row, col in zip(rows, cols)}
    planted = [recovered[i] for i in range(6)] == queries
    if require_planted and not planted:
        raise AssertionError('Planted mapping failed')
    return {'keyword_count': 6, 'documents': 200, 'fixed_sets': 4, 'maximum_cost_error': max_error,
            'exhaustive_assignment_check': True, 'planted_mapping_recovered': planted,
            'reference_mismatch_resolved': False,
            'scope': 'Independent SciPy binomial likelihood calculation and exhaustive assignment on a planted fixture. Full Enron numerical equivalence was not established.'}


def diagnostics(config, stemmers):
    original = read('results/corpus/production-fixed-split-experiments.json')
    coverage = []
    for distribution in ['uniform_iid', 'zipf_iid']:
        counts = [r['scores'][-1]['observed_distinct_keyword_count'] for r in original['runs'] if r['distribution'] == distribution and r['dataset_id'] == 'en:tokenisation_only']
        coverage.append({'distribution': distribution, 'minimum': min(counts), 'maximum': max(counts), 'mean': float(np.mean(counts))})
    vocab_stats = []
    for language in ['en', 'lus']:
        payloads = {}
        for condition in CONDITIONS:
            with (ROOT / f'results/corpus/ihop/{language}-{condition}.pkl').open('rb') as f:
                payloads[condition] = pickle.load(f)
        t, s, st = [set(payloads[c][1]) for c in CONDITIONS]
        vocab = read('results/corpus/wmt23-stopword-removal/vocabulary.json')
        docs = read('results/corpus/wmt23-stopword-removal/documents.json')
        mat = sparse.load_npz(ROOT / 'results/corpus/wmt23-stopword-removal/incidence.npz').tocsr()
        mask = [i for i, d in enumerate(docs) if d['language'] == language]
        freq = np.asarray(mat[mask].sum(axis=0)).ravel()
        groups = defaultdict(list)
        for v, count in zip(vocab, freq):
            if count:
                groups[stemmers[language](v['token'])].append(v['token'])
        vocab_stats.append({'language': language, 'token_stop_overlap': len(t & s), 'stop_stem_string_overlap': len(s & st),
                            'stop_types': int(np.count_nonzero(freq)), 'stem_types': len(groups),
                            'merged_stem_groups': sum(len(v)>1 for v in groups.values()),
                            'types_in_merged_groups': sum(len(v) for v in groups.values() if len(v)>1)})
    near = []
    docs = read('results/corpus/wmt23-tokenisation-only/documents.json')
    matrix = sparse.load_npz(ROOT / 'results/corpus/wmt23-tokenisation-only/incidence.npz').astype(np.int32).tocsr()
    split = read('results/corpus/wmt23-split.json')
    lookup = {d['record_id']: i for i, d in enumerate(docs)}
    for language in ['en', 'lus']:
        aa = [lookup[i] for i in split['assignments'][language]['auxiliary_record_ids']]
        cc = [lookup[i] for i in split['assignments'][language]['client_record_ids']]
        a, c = matrix[aa], matrix[cc]
        ac = np.asarray(a.sum(axis=1)).ravel()
        ccounts = np.asarray(c.sum(axis=1)).ravel()
        hits, clients = 0, set()
        for start in range(0, len(aa), 64):
            overlap = (a[start:start+64] @ c.T).tocoo()
            union = ac[start+overlap.row] + ccounts[overlap.col] - overlap.data
            values = overlap.data / union
            selected = overlap.col[values >= config['near_duplicate_jaccard']]
            hits += len(selected)
            clients.update(selected.tolist())
        near.append({'language': language, 'threshold': config['near_duplicate_jaccard'], 'cross_partition_pair_count': hits,
                     'client_documents_with_match': len(clients), 'screen': 'Unigram-set Jaccard on complete tokenisation-only vocabulary. Original production partition.'})
    return {'coverage': coverage, 'vocabulary': vocab_stats, 'near_duplicates': near, 'independent_check': independent_check()}


def prepare(config):
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, _, stemmers, _ = load_stemming_condition(ROOT / 'configs/stemming_production.json')
    write(OUT / 'diagnostics.json', diagnostics(config, stemmers))
    export_config = read('configs/ihop_corpus_export_production.json')
    selection = read('results/corpus/wmt23-selection.json')
    tasks = []
    def add(payload, name, family, partition, language, condition, distribution=None):
        path = OUT / (name + '.pkl')
        write_pickle(path, payload)
        for dist in ([distribution] if distribution else ['uniform_iid', 'zipf_iid']):
            for seed in config['attack_seeds']:
                tasks.append({'path': str(path), 'dataset_id': name, 'family': family, 'partition': partition,
                              'language': language, 'condition': condition, 'distribution': dist, 'seed': seed,
                              'sha256': file_sha256(path)})
    for partition in config['partition_seeds']:
        split = build_split(selection, {'config_version': '1.0.0', 'mode': 'disjoint_by_matched_pair', 'auxiliary_fraction': 1/3, 'seed': partition})
        write(OUT / f'split-{partition}.json', split)
        for condition in CONDITIONS:
            for language in ['en', 'lus']:
                payload = build_export(ROOT / export_config['conditions'][condition], condition, language, split, config['production_keywords'], True)
                add(payload, f'partition-{partition}-{language}-{condition}', 'partition', partition, language, condition)
    original_split = read('results/corpus/wmt23-split.json')
    controls = []
    for language in ['en', 'lus']:
        payloads = {c: build_export(ROOT / export_config['conditions'][c], c, language, original_split, 1000000, True) for c in CONDITIONS}
        used_stems, chosen = set(), []
        for word in payloads['stopword_removal'][1]:
            stem = stemmers[language](word)
            if stem in used_stems or stem not in payloads['stemming'][1]:
                continue
            chosen.append(word)
            used_stems.add(stem)
            if len(chosen) == config['controlled_keywords']:
                break
        if len(chosen) != config['controlled_keywords']:
            raise ValueError('Controlled keyword set is too small')
        stems = [stemmers[language](word) for word in chosen]
        raw = restrict(payloads['tokenisation_only'], chosen)
        stopped = restrict(payloads['stopword_removal'], chosen)
        if raw[0] != stopped[0]:
            raise AssertionError('Retained-term stopword matrix changed')
        controls.append({'language': language, 'keyword_count': len(chosen), 'raw_stop_matrices_identical': True,
                         'mapped_changed_labels': sum(a != b for a,b in zip(chosen, stems)), 'mapping': list(zip(chosen, stems))})
        for condition, payload in [('stopword_removal', stopped), ('stemming', restrict(payloads['stemming'], stems, chosen))]:
            add(payload, f'controlled-{language}-{condition}', 'controlled', original_split['seed'], language, condition)
        for condition in CONDITIONS:
            with (ROOT / f'results/corpus/ihop/{language}-{condition}.pkl').open('rb') as f:
                payload = pickle.load(f)
            order = np.random.RandomState(20261007).permutation(len(payload[1])).tolist()
            terms = [payload[1][index] for index in order]
            add(restrict(payload, terms), f'shuffled-{language}-{condition}', 'shuffled', original_split['seed'], language, condition, 'zipf_iid')
    write(OUT / 'controlled-keywords.json', controls)
    for partition in config['partition_seeds']:
        selections = {name: read(f'results/corpus/{name}-length-balanced-selection.json') for name in ['flores200', 'wmt23']}
        count = len(selections['flores200']['pairs'])
        ranked = sorted(range(count), key=lambda i: hashlib.sha256(f"{partition}\n{selections['flores200']['pairs'][i]['length_match_id']}".encode()).hexdigest())
        aux = sorted(ranked[:int(count/3+.5)])
        cli = sorted(ranked[len(aux):])
        for corpus, sel in selections.items():
            ec = read(f'configs/ihop_corpus_export_{corpus}_length_balanced.json')
            split = {'mode': 'linked_disjoint_by_length_match', 'seed': partition, 'assignments': {
                lang: {'auxiliary_record_ids': [sel['pairs'][i][lang] for i in aux], 'client_record_ids': [sel['pairs'][i][lang] for i in cli]} for lang in ['en', 'lus']}}
            write(OUT / f'balanced-split-{partition}-{corpus}.json', split)
            for language in ['en', 'lus']:
                payload = build_export(ROOT / ec['conditions']['tokenisation_only'], 'tokenisation_only', language, split, 500, True)
                add(payload, f'balanced-{partition}-{corpus}-{language}', 'balanced', partition, language, corpus)
    manifest = {'config': config, 'config_sha256': file_sha256(ROOT / 'configs/reviewer_sensitivity.json'), 'tasks': tasks}
    write(OUT / 'manifest.json', manifest)
    return manifest


def worker(task, config):
    cfg = {'attack_mode': 'Vol', 'free_fraction': config['free_fraction'], 'iterations': config['iterations'],
           'iteration_checkpoints': [0, config['iterations']], 'query_count_per_keyword': config['query_count_per_keyword']}
    result = run_one(Path(task['path']), task['dataset_id'], cfg, task['distribution'], task['seed'])
    result['task'] = task
    return result


def key(task):
    return f"{task['dataset_id']}:{task['distribution']}:{task['seed']}"


def run(manifest):
    config = manifest['config']
    checkpoint = OUT / 'checkpoint.json'
    saved = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {'manifest_sha256': file_sha256(OUT / 'manifest.json'), 'runs': {}}
    if saved['manifest_sha256'] != file_sha256(OUT / 'manifest.json'):
        raise ValueError('Checkpoint manifest changed')
    todo = [t for t in manifest['tasks'] if key(t) not in saved['runs']]
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        futures = {pool.submit(worker, t, config): t for t in todo}
        for future in as_completed(futures):
            result = future.result()
            saved['runs'][key(result['task'])] = result
            write(checkpoint, saved)
            print(f"Completed {len(saved['runs'])}/{len(manifest['tasks'])} {key(result['task'])}", flush=True)
    records = list(saved['runs'].values())
    summaries = []
    for family in ['partition', 'controlled', 'shuffled', 'balanced']:
        groups = defaultdict(list)
        for r in records:
            t = r['task']
            if t['family'] == family:
                groups[(t['partition'], t['language'], t['condition'], t['distribution'])].append(r)
        for (partition, language, condition, distribution), values in sorted(groups.items()):
            values = sorted(values, key=lambda r: r['seed'])
            summaries.append({'family': family, 'partition': partition, 'language': language, 'condition': condition, 'distribution': distribution,
                              'seeds': [r['seed'] for r in sorted(values, key=lambda r:r['seed'])],
                              'weighted': float(np.mean([r['scores'][-1]['query_weighted_recovery'] for r in values])),
                              'macro': float(np.mean([r['scores'][-1]['distinct_keyword_macro_recovery'] for r in values]))})
    write(OUT / 'analysis.json', {'status': 'passed', 'run_count': len(records), 'config': config, 'summaries': summaries,
                                  'manifest_sha256': file_sha256(OUT / 'manifest.json'), 'checkpoint_sha256': file_sha256(checkpoint),
                                  'inference': 'Post-review descriptive analyses. No additional confirmatory probability claims.'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    config = read('configs/reviewer_sensitivity.json')
    if args.prepare:
        manifest = prepare(config)
        print(f"Prepared {len(manifest['tasks'])} sensitivity runs")
    else:
        run(read('results/corpus/reviewer-sensitivity/manifest.json'))


if __name__ == '__main__':
    main()
