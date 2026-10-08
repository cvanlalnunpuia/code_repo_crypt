from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'results/corpus/production-pipeline-status.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_status(stage, state, **details):
    record = {'stage': stage, 'state': state,
              'updated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'python_version': platform.python_version(), **details}
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(STATUS)
    print(f'{stage} {state}', flush=True)


def command(arguments):
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main():
    stage = 'execution'
    try:
        write_status(stage, 'running')
        command(['scripts/run_fixed_split_experiments.py',
                 'results/corpus/production-fixed-split-experiments.json',
                 '--config', 'configs/fixed_split_experiments_production.json',
                 '--workers', '6', '--progress', '--checkpoint',
                 'results/corpus/production-recovery-checkpoint.json'])
        stage = 'independent_replay'
        write_status(stage, 'running')
        command(['scripts/run_fixed_split_experiments.py',
                 'results/corpus/production-independent-replay.json',
                 '--config', 'configs/fixed_split_experiments_production.json',
                 '--workers', '6', '--progress', '--checkpoint',
                 'results/corpus/production-independent-replay-checkpoint.json'])
        original = ROOT / 'results/corpus/production-fixed-split-experiments.json'
        replay = ROOT / 'results/corpus/production-independent-replay.json'
        if json.loads(original.read_text(encoding='utf-8')) != json.loads(replay.read_text(encoding='utf-8')):
            raise ValueError('Independent replay differs from execution')
        stage = 'analysis'
        write_status(stage, 'running', replay_matched=True)
        command(['scripts/analyse_production_recovery.py'])
        command(['scripts/validate_production_recovery.py'])
        stage = 'tests'
        write_status(stage, 'running', replay_matched=True)
        command(['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
        write_status('complete', 'passed', replay_matched=True,
                     experiment_sha256=digest(original), replay_sha256=digest(replay),
                     analysis_sha256=digest(ROOT / 'results/corpus/production-recovery-analysis.json'))
    except Exception:
        write_status(stage, 'failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
