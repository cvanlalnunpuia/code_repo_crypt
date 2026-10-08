from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    suite = unittest.TestSuite()
    # Production corpus integration tests require separately reconstructed indexes.
    # These families use synthetic fixtures and recorded derived results.
    for pattern in ['test_clarified_stemmer.py', 'test_experiment_checkpoint.py',
                    'test_export_structure.py', 'test_production_recovery_analysis.py',
                    'test_reviewer_sensitivity.py', 'test_springer_template.py',
                    'test_stemmer_execution.py']:
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
