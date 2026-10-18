from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from export_ihop_corpus import ordered_ids, file_sha256, write_pickle

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def corpus(directory, split, language, partition):
    documents = read(directory+'/documents.json')
    vocabulary = read(directory+'/vocabulary.json')
    matrix = sparse.load_npz(ROOT/directory/'incidence.npz').tocsr()
    by_id = {row['record_id']:index for index, row in enumerate(documents)}
    identifiers = split['assignments'][language][partition+'_record_ids']
    return matrix[[by_id[value] for value in identifiers]], vocabulary


def eligible_terms(auxiliary, vocabulary):
    counts = np.asarray(auxiliary.sum(axis=0)).ravel()
    return {row['token']:int(counts[index]) for index, row in enumerate(vocabulary)
            if 0 < counts[index] < auxiliary.shape[0]}


def select_terms(auxiliary_sources, limit):
    candidates = set.intersection(*(set(source) for source in auxiliary_sources))
    terms = sorted(candidates, key=lambda word:(-sum(source[word] for source in auxiliary_sources), word))[:limit]
    if len(terms) != limit:
        raise ValueError('Insufficient auxiliary-supported keywords')
    return terms


def mapped_documents(matrix, vocabulary, terms):
    positions = {row['token']:index for index, row in enumerate(vocabulary)}
    restricted = sparse.hstack([matrix[:, positions[word]] if word in positions else sparse.csr_matrix((matrix.shape[0], 1))
                                for word in terms], format='csr')
    return [sorted(map(int, restricted.getrow(index).indices)) for index in range(matrix.shape[0])]


def build_controls(output, config):
    tasks, diagnostics, sources = [], [], {}
    def register(directory, split_path):
        for name in [directory+'/documents.json', directory+'/vocabulary.json', directory+'/incidence.npz', split_path]:
            sources[name] = file_sha256(ROOT/name)
    def add(name, family, language, condition, auxiliary, auxiliary_vocabulary, client, client_vocabulary, terms):
        aux_docs = mapped_documents(auxiliary, auxiliary_vocabulary, terms)
        cli_docs = mapped_documents(client, client_vocabulary, terms)
        metadata = dict(fixed_split=dict(mode='auxiliary_selected_candidates',
            auxiliary_indices=list(range(len(aux_docs))), client_indices=list(range(len(aux_docs), len(aux_docs)+len(cli_docs)))),
            keyword_selection=dict(source='auxiliary documents only', client_support_filter=False))
        path = output/(name+'.pkl')
        write_pickle(path, (aux_docs+cli_docs, terms, metadata))
        observed = {value for document in cli_docs for value in document}
        diagnostics.append(dict(dataset_id=name, family=family, language=language, condition=condition,
            keyword_count=len(terms), auxiliary_count=len(aux_docs), client_count=len(cli_docs),
            client_absent_keyword_count=len(terms)-len(observed)))
        for distribution in ['uniform_iid', 'zipf_iid']:
            for seed in config['attack_seeds']:
                tasks.append(dict(path=str(path), dataset_id=name, family=family, partition=20260901,
                    language=language, condition=condition, distribution=distribution, seed=seed,
                    sha256=file_sha256(path), client_support_unrestricted=True))
    for condition in ['tokenisation_only', 'stopword_removal', 'stemming']:
        directory = read('configs/ihop_corpus_export_production.json')['conditions'][condition]
        split_path = 'results/corpus/wmt23-split.json'
        split = read(split_path)
        register(directory, split_path)
        for language in ['en', 'lus']:
            auxiliary, vocab = corpus(directory, split, language, 'auxiliary')
            client, _ = corpus(directory, split, language, 'client')
            terms = select_terms([eligible_terms(auxiliary, vocab)], config['production_keywords'])
            add('auxiliary-only-'+language+'-'+condition, 'auxiliary_only', language, condition,
                auxiliary, vocab, client, vocab, terms)
    for language in ['en', 'lus']:
        matrices = {}
        for name, directory, split_path in [
            ('wmt23', 'results/corpus/wmt23-tokenisation-only', 'results/corpus/wmt23-flores200-matched-split.json'),
            ('flores200', 'results/corpus/flores200-tokenisation-only', 'results/corpus/flores200-split.json')]:
            register(directory, split_path)
            split = read(split_path)
            auxiliary, vocab = corpus(directory, split, language, 'auxiliary')
            matrices[name] = auxiliary, vocab, corpus(directory, split, language, 'client')[0]
        if matrices['wmt23'][0].shape[0] != matrices['flores200'][0].shape[0]:
            raise ValueError('Auxiliary-source controls require equal source sizes')
        terms = select_terms([eligible_terms(matrices[name][0], matrices[name][1]) for name in matrices],
                             config['production_keywords'])
        for name, (auxiliary, vocab, _) in matrices.items():
            add('auxiliary-shift-'+language+'-'+name, 'auxiliary_shift', language, name,
                auxiliary, vocab, matrices['wmt23'][2], matrices['wmt23'][1], terms)
    sources['scripts/auxiliary_controls.py'] = file_sha256(Path(__file__))
    (output/'auxiliary-controls.json').write_text(json.dumps(dict(diagnostics=diagnostics,
        scope='Within-collection candidates were selected only from auxiliary frequencies. Auxiliary-source comparisons used equal auxiliary counts, a shared candidate universe selected from both auxiliary sources, and identical WMT23 client documents. Candidates absent from client documents were retained.'), indent=2)+'\n', encoding='utf-8')
    return tasks, sources
