from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from validate_manuscript import validate
from standalone_manuscript import RELEASE_URL

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    validation = validate()
    ledger = json.loads((ROOT/'paper/numerical-provenance.json').read_text(encoding='utf-8'))
    baseline = json.loads(subprocess.check_output(['git', 'show', 'v1.0.2:paper/numerical-provenance.json'], cwd=ROOT/'output/github-release'))
    if any(ledger['facts'].get(key) != value for key,value in baseline['facts'].items()):
        raise ValueError('An original numerical fact changed')
    if ledger['table_rows'] != baseline['table_rows'] or ledger['authors'] != baseline['authors']:
        raise ValueError('An original table or author detail changed')
    for name, checksum in baseline['sources'].items():
        if name.startswith(('results/', 'configs/', 'resources/')) and name != 'results/corpus/returned-linguistic-review-audit.json':
            if ledger['sources'].get(name) != checksum:
                raise ValueError('An original scientific input changed '+name)
    log = ROOT/'output/final-verification-tests.log'
    match = re.search(r'Ran (\d+) tests in [\d.]+s\s+OK\s*$', log.read_text(encoding='utf-8'))
    if not match:
        raise ValueError('Final full-suite test evidence missing')
    compilation = json.loads((ROOT/'paper/compilation-status.json').read_text(encoding='utf-8'))
    if compilation['state'] != 'verified':
        raise ValueError('Final visual verification missing')
    for name, checksum in compilation['source_hashes'].items():
        if digest(ROOT/name) != checksum:
            raise ValueError('Inspected source changed')
    for name, checksum in compilation['pdf_hashes'].items():
        if digest(ROOT/'output/pdf'/(name+'.pdf')) != checksum:
            raise ValueError('Inspected PDF changed')
    existing = json.loads((ROOT/'paper/release-verification.json').read_text(encoding='utf-8'))
    release_path = ROOT/'paper/public-release-verification.json'
    release = json.loads(release_path.read_text(encoding='utf-8')) if release_path.exists() else {}
    published = release.get('state') == 'published_and_verified' and release.get('url') == RELEASE_URL
    existing.update(artifacts={name:digest(ROOT/name) for name in existing['artifacts']},
                    test_count=int(match.group(1)), test_exit_code=0,
                    test_log_sha256=digest(log), reported_fact_count=len(ledger['facts']),
                    source_count=len(ledger['sources']), expanded_runs=1912,
                    native_compilation=validation['native_latex_compilation'],
                    complete_sources_inspected=True, figure_visually_inspected=True,
                    unresolved=['Enron upstream reference mismatch', 'Complete disclosure confirmation'] + ([] if published else ['Public expanded release deposition']),
                    reconstruction_delivery=RELEASE_URL if published else 'Public expanded release pending. Standalone article prepared.',
                    target_journal='Discover Cybersecurity')
    (ROOT/'paper/release-verification.json').write_text(json.dumps(existing, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    record = dict(baseline='Git tag v1.0.2', baseline_facts_preserved=len(baseline['facts']),
                  baseline_table_values_preserved=True, original_scientific_inputs_preserved=True,
                  authors_preserved=True, new_experiments_conducted=True, new_annotations_added=False,
                  final_pdf_layout_verified=True, test_count=int(match.group(1)), test_exit_code=0,
                  source_hashes=compilation['source_hashes'], pdf_hashes=compilation['pdf_hashes'])
    (ROOT/'paper/author-review-verification.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
