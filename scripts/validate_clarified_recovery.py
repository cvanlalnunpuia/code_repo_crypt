from __future__ import annotations

import hashlib
import json
import zipfile

from prepare_clarified_recovery import ARCHIVE, ROOT


def main():
    reuse = json.loads((ROOT / 'results/corpus/clarified-recovery-reuse.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'results/corpus/ihop/manifest.json').read_text(encoding='utf-8'))
    source = 'corpus/annotations/stemmer-production.reviewed.json'
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in [source, 'resources/stemmers/lus-production.reviewed.json',
                     'resources/stemmers/en-production.reviewed.json',
                     'resources/stopwords/lus-production.reviewed.json',
                     'resources/stopwords/en-production.reviewed.json',
                     'configs/fixed_split_experiments_production.json',
                     'configs/production_recovery_analysis.json', 'results/corpus/wmt23-split.json']:
            if archive.read(name) != (ROOT / name).read_bytes():
                raise ValueError('Protected study input changed ' + name)
        for name in ['production-fixed-split-experiments.json', 'production-independent-replay.json']:
            previous = json.loads(archive.read('results/corpus/' + name))
            current = json.loads((ROOT / 'results/corpus' / name).read_text(encoding='utf-8'))
            previous_runs = {(r['dataset_id'], r['distribution'], r['seed']): r for r in previous['runs']}
            current_runs = {(r['dataset_id'], r['distribution'], r['seed']): r for r in current['runs']}
            if previous_runs.keys() != current_runs.keys() or len(current_runs) != len(current['runs']):
                raise ValueError('Experiment run set changed')
            retained = 0
            for key, run in current_runs.items():
                if run['dataset_sha256'] != manifest['datasets'][run['dataset_id']]['sha256']:
                    raise ValueError('Run export provenance differs')
                if reuse['unchanged_attack_inputs'][run['dataset_id']]:
                    expected = dict(previous_runs[key], dataset_sha256=run['dataset_sha256'])
                    if expected != run:
                        raise ValueError('Retained result differs from its archived measurement')
                    retained += 1
            if retained != reuse['retained_runs_per_checkpoint']:
                raise ValueError('Retained run count differs')
    evidence = json.loads((ROOT / 'results/corpus/linguistic-clarification.json').read_text(encoding='utf-8'))
    if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != evidence['annotation_sha256']:
        raise ValueError('Fixed annotations differ from clarification evidence')
    print('Validated protected inputs, retained results, affected run coverage and provenance')


if __name__ == '__main__':
    main()
