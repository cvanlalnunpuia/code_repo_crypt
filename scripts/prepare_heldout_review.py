from __future__ import annotations

import hashlib
import json
from pathlib import Path

from corpus_stemming import load_stemming_condition

ROOT = Path(__file__).resolve().parents[1]


def main():
    _, _, _, stemmers, _ = load_stemming_condition(ROOT / 'configs/stemming_production.json')
    previous = json.loads((ROOT / 'corpus/annotations/stemmer-production.reviewed.json').read_text(encoding='utf-8'))
    excluded = {x['token'] for x in previous['items'] if x['language'] == 'lus'}
    vocabulary = json.loads((ROOT / 'results/corpus/wmt23-stopword-removal/vocabulary.json').read_text(encoding='utf-8'))
    documents = json.loads((ROOT / 'results/corpus/wmt23-stopword-removal/documents.json').read_text(encoding='utf-8'))
    from scipy import sparse
    import numpy as np
    matrix = sparse.load_npz(ROOT / 'results/corpus/wmt23-stopword-removal/incidence.npz').tocsr()
    frequency = np.asarray(matrix[[i for i,d in enumerate(documents) if d['language']=='lus']].sum(axis=0)).ravel()
    groups = {'changed': [], 'unchanged': []}
    for item, count in zip(vocabulary, frequency):
        word = item['token']
        if count and word not in excluded and len(word) >= 4 and word.isalpha():
            groups['changed' if stemmers['lus'](word)!=word else 'unchanged'].append(word)
    chosen = []
    for group, words in groups.items():
        chosen.extend(sorted(words, key=lambda word:hashlib.sha256(('20261007\n'+word).encode()).hexdigest())[:20])
    chosen.sort(key=lambda word:hashlib.sha256(('blind\n'+word).encode()).hexdigest())
    sample = {'language': 'lus', 'sampling_seed': 20261007, 'sampling': 'Twenty changed and twenty unchanged types. Previous annotation tokens were excluded. Hash ranking was applied within each stratum. The combined order was independently hash-ranked.',
              'rule_sha256': hashlib.sha256((ROOT / 'resources/stemmers/lus-production.reviewed.json').read_bytes()).hexdigest(),
              'execution_sha256': hashlib.sha256((ROOT / 'configs/mizo_stemmer_execution.json').read_bytes()).hexdigest(),
              'items': [{'item': i+1, 'token': word, 'expected_stem': '', 'reason': ''} for i,word in enumerate(chosen)]}
    output = ROOT / 'output/heldout-mizo-review'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'annotations.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    instructions = ['# Linguistic review', '', 'Provide the expected stem for each token. Retain the token when no suffix should be removed. Add a short reason for ambiguous or lexicalised forms. Do not consult the stemmer output while completing the review.', '',
                    'The sample contains tokens excluded from the earlier annotation list. The frozen procedure will be scored after the completed annotations are returned. Sample agreement will be reported separately for changed and unchanged forms.', '',
                    '| Item | Token | Expected stem | Reason |', '|---|---|---|---|']
    instructions.extend(f"| {x['item']} | {x['token']} | | |" for x in sample['items'])
    (output / 'review.md').write_text('\n'.join(instructions)+'\n', encoding='utf-8')
    predictions = [{'token': word, 'prediction': stemmers['lus'](word)} for word in chosen]
    (ROOT / 'results/corpus/reviewer-sensitivity/heldout-predictions.json').write_text(json.dumps(predictions, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Prepared {len(chosen)} blinded annotation items')


if __name__ == '__main__':
    main()
