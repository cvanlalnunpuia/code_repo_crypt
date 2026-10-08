from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/corpus/reviewer-sensitivity'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(stage, state, **extra):
    (OUT / 'revision-status.json').write_text(json.dumps({'stage': stage, 'state': state, **extra}, indent=2, sort_keys=True)+'\n', encoding='utf-8')


def command(args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def verify_sensitivity():
    from reviewer_sensitivity import key
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    checkpoint = json.loads((OUT / 'checkpoint.json').read_text(encoding='utf-8'))
    analysis = json.loads((OUT / 'analysis.json').read_text(encoding='utf-8'))
    if set(checkpoint['runs']) != {key(t) for t in manifest['tasks']}:
        raise ValueError('Sensitivity tasks are incomplete')
    if checkpoint['manifest_sha256'] != digest(OUT / 'manifest.json'):
        raise ValueError('Manifest changed')
    if analysis['checkpoint_sha256'] != digest(OUT / 'checkpoint.json'):
        raise ValueError('Sensitivity checkpoint changed')
    for task in manifest['tasks']:
        if digest(Path(task['path'])) != task['sha256']:
            raise ValueError('Sensitivity input changed')
    import numpy as np
    for group in analysis['summaries']:
        runs = [r for r in checkpoint['runs'].values() if all(r['task'][name] == group[name] for name in ['family','partition','language','condition','distribution'])]
        runs = sorted(runs, key=lambda r: r['seed'])
        if len(runs) != len(manifest['config']['attack_seeds']):
            raise ValueError('Sensitivity seed count differs')
        for name, metric in [('weighted','query_weighted_recovery'),('macro','distinct_keyword_macro_recovery')]:
            expected = float(np.mean([r['scores'][-1][metric] for r in runs]))
            if group[name] != expected:
                raise ValueError('Sensitivity mean differs')
    return analysis


def main():
    try:
        status('validation', 'running')
        analysis = verify_sensitivity()
        status('tests', 'running')
        with (OUT / 'tests.stdout.log').open('w', encoding='utf-8') as stdout, (OUT / 'tests.stderr.log').open('w', encoding='utf-8') as stderr:
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'], cwd=ROOT, check=True, stdout=stdout, stderr=stderr)
        tests = (OUT / 'tests.stderr.log').read_text(encoding='utf-8')
        match = re.search(r'Ran (\d+) tests in [\d.]+s\s+OK\s*$', tests)
        if not match:
            raise ValueError('Final tests did not pass')
        status('manuscript', 'running')
        command(['scripts/review_uncertainty.py'])
        command(['scripts/check_real_assignment.py'])
        command(['scripts/check_iterative_assignment.py'])
        command(['scripts/build_manuscript.py'])
        command(['scripts/validate_manuscript.py'])
        from analyse_production_recovery import figure
        figure(ROOT, json.loads((ROOT / 'results/corpus/production-recovery-analysis.json').read_text(encoding='utf-8')))
        ledger = json.loads((ROOT / 'paper/numerical-provenance.json').read_text(encoding='utf-8'))
        for name, sha in ledger['sources'].items():
            if digest(ROOT / name) != sha:
                raise ValueError('Manuscript source changed ' + name)
        release = {'reviewer_sensitivity_runs': analysis['run_count'], 'test_count': int(match.group(1)),
                   'source_count': len(ledger['sources']), 'reported_fact_count': len(ledger['facts']),
                   'native_compilation': 'unverified, built-in compiler check pending',
                   'unresolved': ['Enron upstream reference mismatch', 'Production-scale independent attack equivalence', 'Author declarations and approval', 'Compilation and final PDF layout'],
                   'artifacts': {name:digest(ROOT / name) for name in ['paper/manuscript.tex','paper/supplement.tex','figures/production-recovery.png','figures/partition-sensitivity.png','figures/Fig1.eps','figures/Fig2.eps','paper/validation.json','paper/sn-jnl.cls','paper/sn-mathphys-num.bst','paper/sn-bibliography.bib','paper/Fig1.pdf','paper/Fig2.pdf']}}
        (ROOT / 'paper/release-verification.json').write_text(json.dumps(release, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        command(['scripts/package_manuscript_review.py'])
        command(['scripts/package_overleaf.py'])
        status('compiled_review_pending', 'numerically_verified', run_count=analysis['run_count'], test_count=int(match.group(1)),
               manuscript_sha256=digest(ROOT / 'paper/manuscript.tex'), supplement_sha256=digest(ROOT / 'paper/supplement.tex'))
    except Exception:
        status('revision', 'failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
