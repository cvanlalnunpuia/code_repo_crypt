from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from pathlib import Path

from run_fixed_split_experiments import ROOT

ARCHIVE = ROOT / 'archives/before-linguistic-clarification-20261007/baseline.zip'
CONFIG = ROOT / 'configs/fixed_split_experiments_production.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    manifest_path = ROOT / config['manifest']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    identity = {'config_sha256': digest(CONFIG), 'manifest_sha256': digest(manifest_path),
                'upstream_revision': (ROOT / 'vendor/ihop-code.commit').read_text(encoding='utf-8').strip()}
    reuse = {}
    retained_count = None
    with zipfile.ZipFile(ARCHIVE) as archive:
        old_manifest = json.loads(archive.read(config['manifest']))
        old_config = json.loads(archive.read('configs/fixed_split_experiments_production.json'))
        if old_config != config:
            raise ValueError('Experimental settings changed')
        for dataset_id, entry in manifest['datasets'].items():
            old_entry = old_manifest['datasets'][dataset_id]
            relative = (manifest_path.parent / entry['path']).relative_to(ROOT).as_posix()
            previous = pickle.loads(archive.read(relative))
            with (ROOT / relative).open('rb') as stream:
                current = pickle.load(stream)
            reuse[dataset_id] = (previous[0] == current[0] and previous[1] == current[1]
                                 and previous[2]['fixed_split'] == current[2]['fixed_split'])
            if digest(ROOT / relative) != entry['sha256']:
                raise ValueError('Export digest mismatch')
        if [key for key in reuse if not reuse[key]] != ['lus:stemming']:
            raise ValueError(f'Unexpected affected indexes {reuse}')
        for suffix, original in [('execution', 'production-fixed-split-experiments.json'),
                                 ('replay', 'production-independent-replay.json')]:
            checkpoint_path = ROOT / f'results/corpus/production-clarified-{suffix}-checkpoint.json'
            if checkpoint_path.exists():
                if json.loads(checkpoint_path.read_text(encoding='utf-8'))['identity'] != identity:
                    raise ValueError('Clarified checkpoint provenance differs')
                retained_count = sum(reuse.values()) * len(config['seeds']) * len(config['query_distributions'])
                continue
            original_result = json.loads(archive.read('results/corpus/' + original))
            retained = []
            for run in original_result['runs']:
                if run['dataset_sha256'] != old_manifest['datasets'][run['dataset_id']]['sha256']:
                    raise ValueError('Archived run provenance differs')
                if reuse[run['dataset_id']]:
                    retained.append(dict(run, dataset_sha256=manifest['datasets'][run['dataset_id']]['sha256']))
            checkpoint_path.write_text(json.dumps({'identity': identity, 'runs': retained}, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
            retained_count = len(retained)
    result = {'unchanged_attack_inputs': reuse, 'retained_runs_per_checkpoint': retained_count,
              'method': 'Documents, keyword labels and fixed partition indices were compared exactly with the archived inputs. Unchanged execution and independent-replay runs were retained separately.',
              'identity': identity, 'archive_sha256': digest(ARCHIVE)}
    (ROOT / 'results/corpus/clarified-recovery-reuse.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    prepare()
