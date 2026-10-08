from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    status = json.loads((ROOT / 'results/corpus/production-pipeline-status.json').read_text(encoding='utf-8'))
    if status['state'] != 'passed' or not status['replay_matched']:
        raise ValueError('Clarified pipeline did not pass')
    log = (ROOT / 'results/corpus/clarified-pipeline.stderr.log').read_text(encoding='utf-8')
    match = re.search(r'Ran (\d+) tests in [\d.]+s\s+OK\s*$', log)
    if match is None:
        raise ValueError('Final test result is absent')
    ledger = json.loads((ROOT / 'paper/numerical-provenance.json').read_text(encoding='utf-8'))
    for name, checksum in ledger['sources'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Release source changed ' + name)
    baseline = ROOT / 'archives/before-linguistic-clarification-20261007/baseline.zip'
    with zipfile.ZipFile(baseline) as archive:
        if archive.testzip() is not None:
            raise ValueError('Baseline archive failed integrity check')
    result = {'pipeline': status, 'test_count': int(match.group(1)), 'test_log_sha256': hashlib.sha256(log.encode('utf-8')).hexdigest(),
              'source_count': len(ledger['sources']), 'reported_fact_count': len(ledger['facts']),
              'baseline_preserved': True, 'native_compilation': 'unverified, compiler runtime unavailable',
              'artifacts': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                            ['paper/manuscript.tex', 'paper/supplement.tex', 'figures/production-recovery.png', 'paper/validation.json']}}
    (ROOT / 'paper/release-verification.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('Verified clarified release, source identifiers and final test result')


if __name__ == '__main__':
    main()
