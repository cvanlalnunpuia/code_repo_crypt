from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ledger = json.loads((ROOT / 'paper/numerical-provenance.json').read_text(encoding='utf-8'))
    names = set(ledger['sources'])
    names.update(['paper/manuscript.tex', 'paper/supplement.tex', 'paper/numerical-provenance.json',
                  'paper/validation.json', 'paper/release-verification.json', 'paper/review-checklist.md', 'paper/README.md',
                  'figures/production-recovery.png', 'figures/partition-sensitivity.png', 'figures/Fig1.eps','figures/Fig2.eps',
                  'scripts/build_manuscript.py', 'scripts/validate_manuscript.py', 'scripts/package_manuscript_review.py',
                  'scripts/corpus_stemming.py', 'scripts/analyse_stemmer_execution.py',
                  'scripts/prepare_clarified_recovery.py', 'scripts/complete_clarified_recovery.py',
                  'scripts/validate_clarified_recovery.py',
                  'configs/mizo_stemmer_execution.json'])
    names.update(['paper/reproduction.md', 'paper/compilation-status.json', 'configs/reviewer_sensitivity.json', 'docs/corpus-acquisition.md', 'docs/reviewer-revision.md', 'docs/reviewer-assessment.md',
                  'scripts/reviewer_sensitivity.py', 'scripts/finalize_reviewer_revision.py',
                  'scripts/prepare_heldout_review.py', 'tests/test_reviewer_sensitivity.py', 'requirements-2022.lock'])
    names.update(['paper/submission-information.json', 'paper/submission-readiness.json', 'paper/submission-checklist.md', 'paper/cover-letter.md'])
    names.update(['paper/sn-jnl.cls', 'paper/sn-mathphys-num.bst', 'paper/sn-bibliography.bib', 'paper/Fig1.pdf', 'paper/Fig2.pdf'])
    names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / 'scripts').glob('*.py'))
    names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / 'configs').glob('*.json'))
    names.update(['results/corpus/reviewer-sensitivity/manifest.json', 'results/corpus/reviewer-sensitivity/checkpoint.json'])
    names.update(str(path.relative_to(ROOT)).replace('\\', '/') for folder in ['resources/stopwords', 'resources/stemmers']
                 for path in (ROOT / folder).glob('*production.reviewed.json'))
    manifest = {}
    if 'results/corpus/linguistic-clarification.json' in names:
        raise ValueError('Private clarification text entered the release source ledger')
    for name in sorted(names):
        path = ROOT / name
        if path.suffix.lower() not in {'.json', '.md', '.tex', '.py', '.pdf', '.png', '.eps', '.lock', '.cls', '.bst', '.bib'}:
            raise ValueError('Unexpected review package content ' + name)
        if name in ledger['sources'] and hashlib.sha256(path.read_bytes()).hexdigest() != ledger['sources'][name]:
            raise ValueError('Source changed since manuscript generation ' + name)
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = ROOT / 'output/manuscript-review.zip'
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            archive.write(ROOT / name, name)
        archive.writestr('package-manifest.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Review archive integrity failure')
        for name, checksum in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise ValueError('Packaged file digest mismatch ' + name)
    print(f'Verified review package with {len(names)} files')


if __name__ == '__main__':
    main()
