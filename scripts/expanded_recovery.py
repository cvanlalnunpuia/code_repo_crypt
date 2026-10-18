from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np

from reviewer_sensitivity import ROOT, prepare, run, key, write, read
from export_ihop_corpus import file_sha256
from independent_ihop import run_independent
from audit_production_ihop import run_audited

OUT = ROOT/'results/corpus/expanded-recovery'
CONFIG = ROOT/'configs/expanded_recovery.json'


def expanded_worker(task, config):
    if task.get('implementation') == 'audited':
        cfg = dict(attack_mode='Vol', free_fraction=config['free_fraction'], iterations=config['iterations'],
                   iteration_checkpoints=task['checkpoints'], query_count_per_keyword=config['query_count_per_keyword'])
        result = run_audited(Path(task['path']), task['dataset_id'], cfg, task['distribution'], task['seed'])
        result['task'] = task
        return result
    if task.get('client_support_unrestricted'):
        from run_fixed_split_experiments import run_one
        cfg = dict(attack_mode='Vol', free_fraction=config['free_fraction'], iterations=config['iterations'],
                   iteration_checkpoints=[0, config['iterations']], query_count_per_keyword=config['query_count_per_keyword'])
        result = run_one(Path(task['path']), task['dataset_id'], cfg, task['distribution'], task['seed'], require_client_support=False)
        result['task'] = task
        return result
    if task.get('implementation') != 'independent':
        from reviewer_sensitivity import worker
        return worker(task, config)
    cfg = dict(free_fraction=config['free_fraction'], iterations=config['iterations'],
               iteration_checkpoints=task.get('checkpoints', [0, config['iterations']]), query_count_per_keyword=config['query_count_per_keyword'])
    result = run_independent(Path(task['path']), task['dataset_id'], cfg, task['distribution'], task['seed'])
    result['task'] = task
    return result


def extend_independent():
    manifest = read((OUT/'manifest.json').relative_to(ROOT))
    checkpoint = read((OUT/'checkpoint.json').relative_to(ROOT))
    if any(not item.get('reused_from_previous_execution') for item in checkpoint['runs'].values()):
        raise ValueError('Independent validation must be specified before new recovery runs')
    if any(task.get('implementation') == 'independent' for task in manifest['tasks']):
        raise ValueError('Independent validation is already specified')
    additions = []
    baseline = read('results/corpus/production-fixed-split-experiments.json')
    seeds = {item['seed'] for item in baseline['runs']}
    checkpoints = baseline['runs'][0]['checkpoints']
    for task in manifest['tasks']:
        if task['family'] == 'original_extended' and task['seed'] in seeds:
            additions.append(dict(task, dataset_id='independent-'+task['dataset_id'],
                                  original_dataset_id=task['dataset_id'], family='independent', implementation='independent', checkpoints=checkpoints))
    manifest['tasks'].extend(additions)
    manifest['input_sources']['scripts/independent_ihop.py'] = file_sha256(ROOT/'scripts/independent_ihop.py')
    manifest['input_sources']['scripts/run_fixed_split_experiments.py'] = file_sha256(ROOT/'scripts/run_fixed_split_experiments.py')
    write(OUT/'manifest.json', manifest)
    checkpoint['manifest_sha256'] = file_sha256(OUT/'manifest.json')
    write(OUT/'checkpoint.json', checkpoint)
    write(OUT/'status.json', dict(stage='prepared', task_count=len(manifest['tasks']),
         reused_count=len(checkpoint['runs']), remaining_count=len(manifest['tasks'])-len(checkpoint['runs'])))


def extend_attacker_controls():
    from auxiliary_controls import build_controls
    manifest = read((OUT/'manifest.json').relative_to(ROOT))
    checkpoint = read((OUT/'checkpoint.json').relative_to(ROOT))
    if any(not item.get('reused_from_previous_execution') for item in checkpoint['runs'].values()):
        raise ValueError('Attacker controls must be specified before new recovery runs')
    if any(task['family'] in ['auxiliary_only', 'auxiliary_shift'] for task in manifest['tasks']):
        raise ValueError('Attacker controls already specified')
    tasks, sources = build_controls(OUT, manifest['config'])
    manifest['tasks'].extend(tasks)
    manifest['input_sources'].update(sources)
    manifest['input_sources']['scripts/run_fixed_split_experiments.py'] = file_sha256(ROOT/'scripts/run_fixed_split_experiments.py')
    write(OUT/'manifest.json', manifest)
    checkpoint['manifest_sha256'] = file_sha256(OUT/'manifest.json')
    write(OUT/'checkpoint.json', checkpoint)
    write(OUT/'status.json', dict(stage='prepared', task_count=len(manifest['tasks']),
         reused_count=len(checkpoint['runs']), remaining_count=len(manifest['tasks'])-len(checkpoint['runs'])))


def specify_cost_audit():
    manifest = read((OUT/'manifest.json').relative_to(ROOT))
    checkpoint = read((OUT/'checkpoint.json').relative_to(ROOT))
    if any(not item.get('reused_from_previous_execution') for item in checkpoint['runs'].values()):
        raise ValueError('Audit must be specified before additional results')
    trajectories = []
    for task in manifest['tasks']:
        if task['family'] == 'independent' and task.get('implementation') == 'independent':
            if task['seed'] == 0:
                trajectories.append(dict(task, dataset_id='trajectory-'+task['dataset_id'], family='independent_trajectory'))
            task['implementation'] = 'audited'
    manifest['tasks'].extend(trajectories)
    manifest['input_sources']['scripts/audit_production_ihop.py'] = file_sha256(ROOT/'scripts/audit_production_ihop.py')
    manifest['input_sources']['scripts/expanded_statistics.py'] = file_sha256(ROOT/'scripts/expanded_statistics.py')
    manifest['input_sources']['scripts/expanded_recovery.py'] = file_sha256(Path(__file__))
    manifest['validation_scope'] = 'All original production runs received independent likelihood and assignment-objective checks. Separate trajectories used a distinct implementation on the first original seed. Floating-point tie choices were not required to agree.'
    write(OUT/'manifest.json', manifest)
    checkpoint['manifest_sha256'] = file_sha256(OUT/'manifest.json')
    write(OUT/'checkpoint.json', checkpoint)
    write(OUT/'status.json', dict(stage='prepared', task_count=len(manifest['tasks']),
         reused_count=len(checkpoint['runs']), remaining_count=len(manifest['tasks'])-len(checkpoint['runs'])))


def prepare_expansion():
    if (OUT/'manifest.json').exists() or (OUT/'checkpoint.json').exists():
        raise ValueError('Prepared expansion exists. Resume it without --prepare')
    config = read(CONFIG.relative_to(ROOT))
    old_manifest = read('results/corpus/reviewer-sensitivity/manifest.json')
    settings = ['production_keywords', 'controlled_keywords', 'iterations', 'free_fraction', 'query_count_per_keyword']
    if any(config[name] != old_manifest['config'][name] for name in settings):
        raise ValueError('The expanded study changed an original attack setting')
    manifest = prepare(config, OUT, CONFIG)
    production = read('results/corpus/production-fixed-split-experiments.json')
    for dataset_id in sorted({item['dataset_id'] for item in production['runs']}):
        language, condition = dataset_id.split(':')
        path = ROOT/f'results/corpus/ihop/{language}-{condition}.pkl'
        for seed in config['original_partition_attack_seeds']:
            manifest['tasks'].append(dict(path=str(path), dataset_id=dataset_id,
                family='original_extended', partition=20260901, language=language, condition=condition,
                distribution='uniform_iid', seed=seed, sha256=file_sha256(path)))
            manifest['tasks'].append(dict(manifest['tasks'][-1], distribution='zipf_iid'))
    manifest['input_sources'] = {name:file_sha256(ROOT/name) for name in [
        'configs/expanded_recovery.json', 'configs/reviewer_sensitivity.json',
        'results/corpus/reviewer-sensitivity/manifest.json',
        'results/corpus/reviewer-sensitivity/checkpoint.json',
        'results/corpus/production-fixed-split-experiments.json',
        'vendor/ihop-code/attacks/ihop.py', 'scripts/run_fixed_split_experiments.py']}
    write(OUT/'manifest.json', manifest)
    previous = read('results/corpus/reviewer-sensitivity/checkpoint.json')['runs']
    source_runs = list(previous.values()) + production['runs']
    lookup = {(r['dataset_id'], r['distribution'], r['seed']):r for r in source_runs}
    saved = {'manifest_sha256':file_sha256(OUT/'manifest.json'), 'runs':{}}
    for task in manifest['tasks']:
        existing = lookup.get((task['dataset_id'], task['distribution'], task['seed']))
        if existing is None:
            continue
        if existing['dataset_sha256'] != task['sha256']:
            raise ValueError('Cached dataset differs ' + key(task))
        if existing['checkpoints'][-1] != config['iterations']:
            raise ValueError('Cached endpoint differs')
        cached = deepcopy(existing)
        cached['task'] = task
        cached['reused_from_previous_execution'] = True
        saved['runs'][key(task)] = cached
    write(OUT/'checkpoint.json', saved)
    write(OUT/'status.json', dict(stage='prepared', task_count=len(manifest['tasks']),
                                reused_count=len(saved['runs']), remaining_count=len(manifest['tasks'])-len(saved['runs'])))
    print(json.dumps(read((OUT/'status.json').relative_to(ROOT))), flush=True)


def summarize():
    manifest = read((OUT/'manifest.json').relative_to(ROOT))
    saved = read((OUT/'checkpoint.json').relative_to(ROOT))
    if set(saved['runs']) != {key(task) for task in manifest['tasks']}:
        raise ValueError('The expansion is incomplete')
    groups = defaultdict(list)
    for record in saved['runs'].values():
        task = record['task']
        groups[tuple(task[name] for name in ['family', 'partition', 'language', 'condition', 'distribution'])].append(record)
    summaries = []
    for group, records in sorted(groups.items()):
        records.sort(key=lambda value:value['seed'])
        row = dict(zip(['family', 'partition', 'language', 'condition', 'distribution'], group))
        row['seeds'] = [r['seed'] for r in records]
        expected = ([0] if row['family']=='independent_trajectory' else
                    manifest['config']['original_partition_attack_seeds' if row['family']=='original_extended' else 'attack_seeds'])
        if row['seeds'] != expected:
            raise ValueError('A summary has missing seeds')
        for metric in ['query_weighted_recovery', 'distinct_keyword_macro_recovery']:
            values = [r['scores'][-1][metric] for r in records]
            row[metric] = dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)))
        summaries.append(row)
    write(OUT/'expanded-analysis.json', dict(status='passed', config=manifest['config'],
        run_count=len(saved['runs']), summaries=summaries, checkpoint_sha256=file_sha256(OUT/'checkpoint.json'),
        interpretation='Additional descriptive results. Original confirmatory comparisons were preserved.'))
    from expanded_statistics import original_extension_statistics
    write(OUT/'extended-statistics.json', original_extension_statistics(list(saved['runs'].values()), manifest['config']))
    baseline = read('results/corpus/production-fixed-split-experiments.json')
    expected = {(r['dataset_id'], r['distribution'], r['seed']):r for r in baseline['runs']}
    comparisons = []
    for r in saved['runs'].values():
        task = r['task']
        if task['family'] != 'independent':
            continue
        reference = expected[(task['original_dataset_id'], task['distribution'], task['seed'])]
        if r['checkpoints'] != reference['checkpoints']:
            raise ValueError('Independent checkpoint coverage differs')
        errors = [abs(a[metric]-b[metric]) for a, b in zip(r['scores'], reference['scores'])
                  for metric in ['query_weighted_recovery', 'distinct_keyword_macro_recovery']]
        comparisons.append(dict(dataset_id=task['original_dataset_id'], distribution=task['distribution'],
                                seed=task['seed'], maximum_score_difference=max(errors), matched=max(errors)<=1e-12,
                                cost_audit=r['cost_audit']))
    write(OUT/'independent-comparison.json', dict(comparisons=comparisons,
        all_matched=all(row['matched'] for row in comparisons), comparison_count=len(comparisons),
        tolerance=1e-12, scope='Independent likelihood and assignment-objective checks on every original production run. Recovery scores were compared at every original checkpoint. The SciPy assignment library was shared. Independent trajectory identity was not required.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--extend-independent', action='store_true')
    parser.add_argument('--extend-attacker-controls', action='store_true')
    parser.add_argument('--specify-cost-audit', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare_expansion()
        return
    if args.extend_independent:
        extend_independent()
        return
    if args.extend_attacker_controls:
        extend_attacker_controls()
        return
    if args.specify_cost_audit:
        specify_cost_audit()
        return
    manifest = read((OUT/'manifest.json').relative_to(ROOT))
    for name, expected in manifest['input_sources'].items():
        if file_sha256(ROOT/name) != expected:
            raise ValueError('A frozen input changed ' + name)
    for task in manifest['tasks']:
        if file_sha256(Path(task['path'])) != task['sha256']:
            raise ValueError('An exported task input changed')
    write(OUT/'status.json', dict(stage='running', task_count=len(manifest['tasks']), process_id=os.getpid()))
    try:
        run(manifest, OUT, expanded_worker)
        summarize()
    except BaseException as error:
        write(OUT/'status.json', dict(stage='failed', task_count=len(manifest['tasks']),
                                    process_id=os.getpid(), error=str(error)))
        raise
    write(OUT/'status.json', dict(stage='complete', task_count=len(manifest['tasks'])))


if __name__ == '__main__':
    main()
