from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build_import(root):
    audit = load(root / 'results/corpus/returned-linguistic-review-audit.json')
    source = Path(audit['source_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != audit['source_sha256']:
        raise ValueError('Returned workbook changed after audit')
    assignment = load(root / 'configs/linguistic_review.json')
    if audit['reviewer'] != assignment['reviewer']['name'] or audit['issues']:
        raise ValueError('Reviewer identity or workbook validation remains unresolved')
    date.fromisoformat(audit['completion_date'])
    reviewer = audit['reviewer']
    reviewed_at = audit['completion_date']
    outputs = {}
    for language in ('en', 'lus'):
        payload = load(root / f'resources/stopwords/{language}-production.pending.json')
        decisions = {item['token']: item for item in audit['stopwords'][language]['entries']}
        if set(decisions) != {item['token'] for item in payload['entries']}:
            raise ValueError('Stopword identities differ')
        for entry in payload['entries']:
            decision = decisions[entry['token']]
            entry['decision'] = decision['decision']
            if entry['decision'] not in {'retain', 'exclude_from_index'}:
                raise ValueError('Incomplete stopword decision')
            if decision['reviewer_basis']:
                entry['basis'] = decision['reviewer_basis']
            else:
                entry['basis'] += f". Decision recorded by {reviewer} in {decision['source_cell']} on {reviewed_at}."
        payload.update(status='approved', reviewed_by=reviewer, reviewed_at=reviewed_at, list_version='1.0.0')
        outputs[f'resources/stopwords/{language}-production.reviewed.json'] = payload
    english = load(root / 'resources/stemmers/en-production.pending.json')
    if audit['english_algorithm_decision'] != 'approve':
        raise ValueError('English algorithm lacks approval')
    english.update(status='approved', reviewed_by=reviewer, reviewed_at=reviewed_at, specification_version='1.0.0')
    outputs['resources/stemmers/en-production.reviewed.json'] = english
    mizo = load(root / 'resources/stemmers/lus-production.pending.json')
    mizo.update(status='approved', reviewed_by=reviewer, reviewed_at=reviewed_at, specification_version='1.0.0', rules=audit['mizo_rules'])
    outputs['resources/stemmers/lus-production.reviewed.json'] = mizo
    staged = dict(mizo, status='pending')
    outputs['resources/stemmers/lus-production.pending.json'] = staged
    outputs['corpus/annotations/stemmer-production.reviewed.json'] = {
        'annotation_version': '1.0.0', 'scope': 'production', 'status': 'approved',
        'annotator': reviewer, 'annotated_at': reviewed_at,
        'items': [{'language': language, 'token': item['token'], 'expected_stem': item['expected_stem']} for language in ('en', 'lus') for item in audit['annotations'][language]['items']],
    }
    outputs['results/corpus/linguistic-review-import.json'] = {
        'reviewer': reviewer, 'reviewed_at': reviewed_at,
        'source_path': str(source), 'source_sha256': audit['source_sha256'],
        'audit_sha256': hashlib.sha256((root / 'results/corpus/returned-linguistic-review-audit.json').read_bytes()).hexdigest(),
        'resource_paths': sorted(outputs),
        'stopword_counts': {language: audit['stopwords'][language]['counts'] for language in ('en', 'lus')},
        'mizo_rule_count': len(audit['mizo_rules']),
        'annotation_count': sum(arm['item_count'] for arm in audit['annotations'].values()),
        'mizo_execution_status': 'pending',
        'mizo_execution_requirements': [
            'The accepted in rule requires a reviewed lexicon check.',
            'The review basis for ahte requires sequential te and ah removal.',
            'Annotation disagreements require a documented resolution without treating evaluation labels as exception rules.',
        ],
    }
    return outputs


def main():
    root = Path(__file__).resolve().parents[1]
    outputs = build_import(root)
    print('*** Begin Patch')
    for relative, payload in outputs.items():
        path = root / relative
        content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
        if path.exists() and path.read_text(encoding='utf-8') == content:
            continue
        if path.exists():
            print(f'*** Update File: {path}')
            print('@@')
            for line in path.read_text(encoding='utf-8').splitlines():
                print('-' + line)
        else:
            print(f'*** Add File: {path}')
        for line in content.splitlines():
            print('+' + line)
    print('*** End Patch')


if __name__ == '__main__':
    main()
