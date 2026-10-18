from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm-all-pages-inspected', action='store_true', required=True)
    parser.add_argument('--built-in-verified', action='store_true')
    args = parser.parse_args()
    path = ROOT/'paper/compilation-status.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    bounds = json.loads((ROOT/'paper/pdf-bounds-check.json').read_text(encoding='utf-8'))
    for name, expected in record['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('A source changed after compilation')
    for name, details in bounds.items():
        if hashlib.sha256((ROOT/'output/pdf'/(name+'.pdf')).read_bytes()).hexdigest() != details['sha256']:
            raise ValueError('The inspected PDF changed')
        if details['out_of_page_words']:
            raise ValueError('The PDF bounds check failed')
    record.update(state='verified', visual_inspection='All pages inspected using rendered page images and detailed inspection of key pages',
                  page_counts={name:details['page_count'] for name,details in bounds.items()},
                  built_in_compiler=('Both saved sources compiled successfully with the desktop editor compiler.' if args.built_in_verified
                                     else 'The installed pdfLaTeX compiler was used. Built-in compilation was not verified in this inspection.'),
                  native_editor_compilation='verified' if args.built_in_verified else 'unverified')
    path.write_text(json.dumps(record, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print('Recorded visual verification of the exact compiled source and PDF hashes')


if __name__ == '__main__':
    main()
