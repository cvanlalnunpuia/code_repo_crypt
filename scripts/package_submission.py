from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from standalone_manuscript import RELEASE_URL

ROOT = Path(__file__).resolve().parents[1]


def checked_zip(path, files):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, source in files.items():
            archive.write(source, name)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(files)
        for name, source in files.items():
            assert archive.read(name) == source.read_bytes()
    return {name: hashlib.sha256(source.read_bytes()).hexdigest() for name, source in files.items()}


def main():
    shared = {name: ROOT/'paper'/name for name in ['sn-jnl.cls', 'sn-mathphys-num.bst', 'Fig1.pdf', 'Fig2.pdf']}
    article = dict(shared)
    article.update({'main.tex': ROOT/'paper/manuscript.tex', 'sn-bibliography.bib': ROOT/'paper/sn-bibliography.bib'})
    article.update({name: ROOT/'figures'/name for name in ['Fig1.eps', 'Fig2.eps']})
    supplement = {'supplement.tex': ROOT/'paper/supplement.tex', 'sn-jnl.cls': ROOT/'paper/sn-jnl.cls',
                  'sn-mathphys-num.bst': ROOT/'paper/sn-mathphys-num.bst'}
    out = ROOT/'output'
    out.mkdir(exist_ok=True)
    record = {'article': checked_zip(out/'paper-submission.zip', article),
              'supplement': checked_zip(out/'supplement-source.zip', supplement),
              'supplement_required': False,
              'reconstruction_delivery': RELEASE_URL,
              'final_pdf_verified': json.loads((ROOT/'paper/compilation-status.json').read_text(encoding='utf-8'))['state'] == 'verified'}
    (out/'submission-file-checks.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    (out/'cover-letter.txt').write_bytes((ROOT/'paper/cover-letter.md').read_bytes())
    print('Verified article and supplementary source archives')


if __name__ == '__main__':
    main()
