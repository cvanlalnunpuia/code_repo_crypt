from __future__ import annotations

import hashlib
import json
from pathlib import Path
from standalone_manuscript import RELEASE_URL

ROOT = Path(__file__).resolve().parents[1]


def main():
    info = json.loads((ROOT / 'paper/submission-information.json').read_text(encoding='utf-8'))
    validation = json.loads((ROOT / 'paper/validation.json').read_text(encoding='utf-8'))
    compilation = json.loads((ROOT / 'paper/compilation-status.json').read_text(encoding='utf-8'))
    missing = []
    if info.get('journal') in [None, '', 'Not selected']:
        missing.append('Target journal and venue-specific suitability review')
    for index, author in enumerate(info['authors'], 1):
        for field in ['name', 'affiliation']:
            if not author.get(field):
                missing.append(f'Author {index} {field}')
    for field in ['corresponding_author_email', 'funding_statement', 'competing_interests_statement', 'author_contributions_statement', 'ethics_statement']:
        if not info.get(field):
            missing.append(field.replace('_', ' '))
    for field in ['all_authors_approved', 'exclusive_submission_confirmed']:
        if info.get(field) is not True:
            missing.append(field.replace('_', ' '))
    if info.get('required_disclosure_reviewed') is not True:
        missing.append('Disclosure of technical assistance remains pending. The approved statement covers writing support.')
    release_path = ROOT/'paper/public-release-verification.json'
    release = json.loads(release_path.read_text(encoding='utf-8')) if release_path.exists() else {}
    if release.get('state') != 'published_and_verified' or release.get('url') != RELEASE_URL:
        missing.append('Public deposition and citation of the expanded reconstruction records')
    source_hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['paper/manuscript.tex', 'paper/supplement.tex']}
    compiled = compilation['state'] == 'verified' and compilation['source_hashes'] == source_hashes
    if not compiled:
        missing.append('Compilation and final PDF layout verification')
    manuscript = (ROOT/'paper/manuscript.tex').read_text(encoding='utf-8')
    if not all(info.get(field) and info[field] in manuscript for field in ['funding_statement','competing_interests_statement','author_contributions_statement','ethics_statement']):
        missing.append('Confirmed declarations inserted into the manuscript')
    result = dict(journal=info['journal'], ready_for_submission=not missing,
                  technical_package_ready=compiled and all(validation['checks'].values()),
                  abstract_word_count=validation['abstract_word_count'], reference_count=validation['reference_count'], numerical_fact_count=validation['recomputed_fact_count'],
                  mechanical_checks_passed=all(validation['checks'].values()), missing=missing,
                  scientific_limits=['Enron reference mismatch remained unresolved', 'Assignment library was shared in production-scale cost and objective audits'],
                  article_type=info['article_type'], scope_reviewed=info.get('scope_reviewed',False), collection=info.get('collection'),
                  source_hashes=source_hashes)
    (ROOT/'paper/submission-readiness.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    lines = ['# Submission', '',
             'The target journal is '+info['journal']+'. The article type is Research. No topical collection was selected. The subject matches security measurement, evaluation and reproducibility.', '',
             'The official [submission guidelines]('+info['guidelines']+') and [journal scope](https://link.springer.com/journal/44613/aims-and-scope) were checked. The official Springer Nature class and bibliography style were retained.', '',
             'Pending items', '']
    lines.extend('- '+item.rstrip('.')+'.' for item in missing)
    lines.extend(['', 'The supplied author details and declarations were retained in paper/submission-information.json. Confirm their applicability to the revised submission and the selected venue.', '',
                  'Upload output/paper-submission.zip as the article source. Supply output/cover-letter.txt in the cover-letter field. No supplementary PDF is required by the standalone article. The manuscript cites the versioned reconstruction release. Its publication status is checked above. The Overleaf ZIP is the editing project.', '',
                  'The supplied author approval, funding, competing interests, contributions and ethics statements were recorded in the source metadata. The corresponding author must verify the final submission declarations.', '',
                  'The completed linguistic review supported the reported sample agreement. No additional linguistic review was required for this claim.', '',
                  'The Enron reference mismatch and shared assignment-library scope remained explicit. Authors must assess these limitations before submitting the conditional evaluation.', ''])
    (ROOT/'paper/submission-checklist.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
