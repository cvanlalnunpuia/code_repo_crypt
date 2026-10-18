from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/corpus/expanded-recovery'


def main():
    path = OUT/'expanded-analysis.json'
    original = path.read_bytes()
    analysis = json.loads(original)
    count = 0
    for row in analysis['summaries']:
        for metric in ['query_weighted_recovery', 'distinct_keyword_macro_recovery']:
            value = row[metric]['sd']
            if value is not None and not math.isfinite(value):
                if len(row['seeds']) != 1:
                    raise ValueError('Non-finite standard deviation in a repeated-seed group')
                row[metric]['sd'] = None
                count += 1
    if count:
        path.write_text(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
        record = dict(operation='Singleton sample standard deviations represented as null (unmeasured)',
                      fields_changed=count, original_sha256=hashlib.sha256(original).hexdigest(),
                      normalized_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      experimental_settings_changed=False, saved_runs_changed=False,
                      frozen_execution_sources_changed=False)
        (OUT/'summary-format-provenance.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    json.loads(path.read_text(encoding='utf-8'), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    print('Expanded summary uses finite numbers and unmeasured singleton standard deviations')


if __name__ == '__main__':
    main()
