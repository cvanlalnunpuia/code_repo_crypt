from __future__ import annotations

import json
from pathlib import Path

from corpus_records import read_jsonl
from corpus_tokenisation import file_sha256, tokenise


def build(root):
    records_path = root / 'results/corpus/wmt23-records.jsonl'
    split_path = root / 'results/corpus/wmt23-split.json'
    config_path = root / 'configs/tokenisation_only.json'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    identifiers = set(split['assignments']['lus']['auxiliary_record_ids'])
    tokens = set()
    found = set()
    for record in read_jsonl(records_path):
        if record['record_id'] not in identifiers:
            continue
        if record['language'] != 'lus':
            raise ValueError('Auxiliary Mizo split includes another language')
        found.add(record['record_id'])
        tokens.update(tokenise('\n\n'.join(record[field] for field in config['indexed_fields']), config))
    if found != identifiers:
        raise ValueError('Auxiliary records are missing')
    return {
        'language': 'lus', 'partition': 'auxiliary', 'document_count': len(found),
        'tokens': sorted(tokens), 'token_count': len(tokens),
        'source_sha256': file_sha256(records_path), 'split_sha256': file_sha256(split_path),
        'tokenisation_sha256': file_sha256(config_path),
        'meaning': 'Attested word forms. Membership does not establish a lexical root.',
    }


def main():
    root = Path(__file__).resolve().parents[1]
    result = build(root)
    output = root / 'results/corpus/mizo-stemmer-auxiliary-wordlist.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f"Built {result['token_count']} attested forms from {result['document_count']} auxiliary Mizo documents")


if __name__ == '__main__':
    main()
