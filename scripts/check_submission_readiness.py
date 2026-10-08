from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    info = json.loads((ROOT / 'paper/submission-information.json').read_text(encoding='utf-8'))
    validation = json.loads((ROOT / 'paper/validation.json').read_text(encoding='utf-8'))
    compilation = json.loads((ROOT / 'paper/compilation-status.json').read_text(encoding='utf-8'))
    missing = []
    for index, author in enumerate(info['authors'], 1):
        for field in ['name', 'affiliation']:
            if not author.get(field):
                missing.append(f'Author {index} {field}')
    for field in ['corresponding_author_email', 'funding_statement', 'competing_interests_statement', 'author_contributions_statement', 'ethics_statement']:
        if not info.get(field):
            missing.append(field.replace('_', ' '))
    for field in ['all_authors_approved', 'exclusive_submission_confirmed', 'required_disclosure_reviewed']:
        if info.get(field) is not True:
            missing.append(field.replace('_', ' '))
    source_hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['paper/manuscript.tex', 'paper/supplement.tex']}
    compiled = compilation['state'] == 'verified' and compilation['source_hashes'] == source_hashes
    if not compiled:
        missing.append('Compilation and final PDF layout verification')
    missing.append('Author-approved cover letter and declarations inserted into the manuscript')
    result = dict(journal=info['journal'], ready_for_submission=False,
                  abstract_word_count=validation['abstract_word_count'], reference_count=validation['reference_count'], numerical_fact_count=validation['recomputed_fact_count'],
                  mechanical_checks_passed=all(validation['checks'].values()), missing=missing,
                  scientific_limits=['Enron reference mismatch remained unresolved', 'Production-scale independent attack equivalence remained unvalidated'],
                  source_hashes=source_hashes)
    (ROOT/'paper/submission-readiness.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    lines = ['# Submission', '',
             'The target journal is Discover Informatics. The source package was prepared for a Research article. Submission remained pending author information and final PDF verification.', '',
             'The abstract limit, declarations, cover letter and supplementary-file requirements were checked against the [submission guidelines](https://link.springer.com/journal/44564/submission-guidelines). Assistance disclosure required author review under the [editorial policies](https://link.springer.com/brands/discover/policies).', '',
             'Required author information', '']
    lines.extend('- '+item+'.' for item in missing)
    lines.extend(['', 'Fill paper/submission-information.json with approved details. Include an ethics assessment appropriate to the public secondary data and linguistic review. No exemption, funding status or absence of conflicts was inferred.', '',
                  'Select pdfLaTeX in Overleaf. Compile main.tex and supplement.tex separately. Inspect every page for clipped tables, overlapping labels and missing references. Export the supplementary PDF for upload. Earlier local PDFs were excluded from the updated packages.', '',
                  'The cover letter remained an unsigned draft. Final author approval and submission declarations were required.', '',
                  'The completed linguistic review supported the reported sample agreement. No additional linguistic review was required for this claim.', '',
                  'The Enron reference mismatch and production-scale independent validation limit remained explicit. Authors must assess these limitations before submitting the conditional evaluation.', ''])
    (ROOT/'paper/submission-checklist.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
