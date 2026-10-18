from __future__ import annotations

import argparse
import hashlib
import json
import re
import unittest
from pathlib import Path

from validate_manuscript import validate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-exit-code', type=int, required=True)
    args = parser.parse_args()
    if args.test_exit_code != 0:
        raise ValueError('The complete test suite must pass before recording completion')
    validation = validate()
    current = json.loads((ROOT/'paper/numerical-provenance.json').read_text(encoding='utf-8'))
    baseline_path = ROOT/'output/github-release/paper/numerical-provenance.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if not all(current['facts'][name] == value for name, value in baseline['facts'].items()):
        raise ValueError('A baseline numerical fact changed')
    if current['table_rows'] != baseline['table_rows']:
        raise ValueError('Baseline table values changed')
    if current['authors'] != baseline['authors']:
        raise ValueError('The approved author order or contact details changed')
    for name, checksum in baseline['sources'].items():
        if name.startswith(('results/', 'configs/', 'resources/')) and current['sources'].get(name) != checksum:
            if name != 'results/corpus/returned-linguistic-review-audit.json':
                raise ValueError('An experimental or linguistic source changed ' + name)
            local_audit = json.loads((ROOT/name).read_text(encoding='utf-8'))
            public_audit = json.loads((ROOT/'output/github-release'/name).read_text(encoding='utf-8'))
            local_name = local_audit.pop('source_path').replace('\\', '/').split('/')[-1]
            public_name = public_audit.pop('source_path')
            if local_name != public_name or local_audit != public_audit:
                raise ValueError('Linguistic audit differs beyond the public path sanitisation')
    draft_logs = {}
    for name in ['manuscript', 'supplement']:
        path = ROOT/'tmp/latex-guidelines'/(name+'.log')
        log = path.read_text(encoding='utf-8', errors='replace')
        if re.search(r'^!|Overfull|Float too large|undefined|bookmark levels', log, flags=re.M):
            raise ValueError('Draft compilation requires review ' + name)
        if 'PDF statistics:' not in log or r'\pdfdraftmode enabled' not in log:
            raise ValueError('Draft compilation did not finish ' + name)
        draft_logs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = lambda name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    sources = ['paper/manuscript.tex', 'paper/supplement.tex']
    compilation = {
        'state': 'unverified',
        'reason': 'Built-in compiler cannot access the required project class',
        'diagnostics': 'The built-in compiler reported sn-jnl.cls missing. Local draft-mode TeX and BibTeX checks passed. Final PDF layout remained unverified.',
        'paths': sources,
        'source_hashes': {name:digest(name) for name in sources},
        'draft_log_hashes': draft_logs,
    }
    existing_compilation = json.loads((ROOT/'paper/compilation-status.json').read_text(encoding='utf-8'))
    if existing_compilation.get('state') == 'verified' and existing_compilation.get('source_hashes') == compilation['source_hashes']:
        compilation = existing_compilation
    (ROOT/'paper/compilation-status.json').write_text(json.dumps(compilation, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    test_count = unittest.defaultTestLoader.discover(str(ROOT/'tests')).countTestCases()
    record = {
        'baseline': baseline_path.relative_to(ROOT).as_posix(),
        'baseline_sha256': digest(baseline_path.relative_to(ROOT)),
        'baseline_facts_preserved': len(baseline['facts']),
        'baseline_table_values_preserved': True,
        'experimental_and_linguistic_inputs_preserved': True,
        'public_audit_path_sanitisation_checked': True,
        'authors_preserved': True,
        'recomputed_fact_count': validation['recomputed_fact_count'],
        'reference_count': validation['reference_count'],
        'test_count': test_count,
        'test_exit_code': args.test_exit_code,
        'source_hashes': compilation['source_hashes'],
        'draft_compilation_passed': True,
        'final_pdf_layout_verified': compilation['state'] == 'verified',
        'new_experiments_conducted': False,
        'new_annotations_added': False,
        'public_release_updated': False,
    }
    (ROOT/'paper/author-review-verification.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    release_path = ROOT/'paper/release-verification.json'
    release = json.loads(release_path.read_text(encoding='utf-8'))
    release['artifacts'] = {name:digest(name) for name in release['artifacts']}
    release.update(test_count=test_count, test_exit_code=args.test_exit_code,
                   reported_fact_count=validation['recomputed_fact_count'], source_count=validation['source_count'],
                   native_compilation=validation['native_latex_compilation'])
    release['unresolved'] = ['Enron upstream reference mismatch', 'Production-scale independent attack equivalence',
                             'Target venue selection', 'Author review of revised manuscript and required disclosures',
                             'Compilation and final PDF layout', 'Updated public reconstruction release']
    if compilation['state'] == 'verified':
        release['unresolved'].remove('Compilation and final PDF layout')
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
