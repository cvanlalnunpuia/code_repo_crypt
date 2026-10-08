from __future__ import annotations

import json

from complete_production_recovery import ROOT, command, digest, write_status


def main():
    try:
        for stage, suffix, output in [('execution', 'execution', 'production-fixed-split-experiments.json'),
                                      ('independent_replay', 'replay', 'production-independent-replay.json')]:
            write_status(stage, 'running', execution_policy='reviewer_clarified_repeated_suffix')
            command(['scripts/run_fixed_split_experiments.py', 'results/corpus/' + output,
                     '--config', 'configs/fixed_split_experiments_production.json', '--workers', '4', '--progress',
                     '--checkpoint', f'results/corpus/production-clarified-{suffix}-checkpoint.json'])
        original = ROOT / 'results/corpus/production-fixed-split-experiments.json'
        replay = ROOT / 'results/corpus/production-independent-replay.json'
        if json.loads(original.read_text(encoding='utf-8')) != json.loads(replay.read_text(encoding='utf-8')):
            raise ValueError('Clarified independent replay differs')
        command(['scripts/validate_clarified_recovery.py'])
        write_status('analysis', 'running', replay_matched=True)
        command(['scripts/analyse_production_recovery.py'])
        command(['scripts/validate_production_recovery.py'])
        command(['scripts/analyse_export_structure.py'])
        command(['scripts/validate_export_structure.py'])
        write_status('tests', 'running', replay_matched=True)
        command(['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
        write_status('complete', 'passed', replay_matched=True,
                     execution_policy='reviewer_clarified_repeated_suffix',
                     experiment_sha256=digest(original), replay_sha256=digest(replay),
                     analysis_sha256=digest(ROOT / 'results/corpus/production-recovery-analysis.json'))
    except Exception:
        import traceback
        write_status('clarified_recovery', 'failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
