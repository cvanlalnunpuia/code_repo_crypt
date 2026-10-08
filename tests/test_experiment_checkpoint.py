from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from run_fixed_split_experiments import run_experiments


def synthetic_run(task):
    path, dataset_id, config, distribution, seed = task
    return {'dataset_id': dataset_id, 'distribution': distribution, 'seed': seed,
            'checkpoints': config['iteration_checkpoints'],
            'scores': [{'query_weighted_recovery': 0.25,
                        'distinct_keyword_macro_recovery': 0.5}
                       for _ in config['iteration_checkpoints']]}


class ExperimentCheckpointTests(unittest.TestCase):
    def test_interrupted_run_resumes_to_same_result(self):
        config = ROOT / 'configs/fixed_split_experiments_fixture.json'
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'checkpoint.json'
            call_count = 0

            def interrupted(task):
                nonlocal call_count
                call_count += 1
                if call_count == 3:
                    raise RuntimeError('Synthetic interruption')
                return synthetic_run(task)

            with patch('run_fixed_split_experiments.run_task', interrupted):
                with self.assertRaisesRegex(RuntimeError, 'Synthetic interruption'):
                    run_experiments(config, checkpoint_path=checkpoint)
            self.assertEqual(len(json.loads(checkpoint.read_text())['runs']), 2)
            with patch('run_fixed_split_experiments.run_task', side_effect=synthetic_run) as runner:
                resumed = run_experiments(config, checkpoint_path=checkpoint)
                self.assertEqual(runner.call_count, 34)
                fresh = run_experiments(config)
            self.assertEqual(resumed, fresh)

    def test_checkpoint_rejects_changed_provenance(self):
        config = ROOT / 'configs/fixed_split_experiments_fixture.json'
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'checkpoint.json'
            with patch('run_fixed_split_experiments.run_task', synthetic_run):
                run_experiments(config, checkpoint_path=checkpoint)
            stored = json.loads(checkpoint.read_text())
            stored['identity']['config_sha256'] = 'invalid'
            checkpoint.write_text(json.dumps(stored))
            with self.assertRaisesRegex(ValueError, 'provenance differs'):
                run_experiments(config, checkpoint_path=checkpoint)


if __name__ == '__main__':
    unittest.main()
