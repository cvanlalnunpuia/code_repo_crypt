from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = {'main.tex': ROOT / 'paper/manuscript.tex',
             'supplement.tex': ROOT / 'paper/supplement.tex',
             'figures/production-recovery.png': ROOT / 'figures/production-recovery.png',
             'figures/partition-sensitivity.png': ROOT / 'figures/partition-sensitivity.png',
             'figures/Fig1.eps':ROOT/'figures/Fig1.eps','figures/Fig2.eps':ROOT/'figures/Fig2.eps'}
    files.update({name:ROOT/'paper'/name for name in ['sn-jnl.cls', 'sn-mathphys-num.bst', 'sn-bibliography.bib', 'Fig1.pdf', 'Fig2.pdf']})
    verification = json.loads((ROOT / 'paper/release-verification.json').read_text(encoding='utf-8'))
    for path in files.values():
        relative = path.relative_to(ROOT).as_posix()
        if hashlib.sha256(path.read_bytes()).hexdigest() != verification['artifacts'][relative]:
            raise ValueError('Verified source changed ' + relative)
    guide = ('Overleaf project\n\n'
             'Upload this ZIP as a new Overleaf project.\n'
             'Set main.tex as the main document for the article.\n'
             'Set supplement.tex as the main document to compile the supplementary material separately.\n'
             'Select pdfLaTeX as the compiler.\n\n'
             'Both sources use the official Springer Nature class and its supplied numbered bibliography style. The bibliography data and external vector figures were included. '
             'The PNG versions of the figures are also included.\n\n'
             'The supplied six-author order, affiliations and email addresses were included. C Vanlalnunpuia was designated as corresponding author.\n'
             'The sources contain the clarified linguistic procedure and independently checked recovery results.\n'
             'Compilation remains unverified because the local compiler runtime was unavailable.\n')
    output = ROOT / 'output/paper-overleaf.zip'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr('README.txt', guide)
        for name in ['submission-checklist.md', 'cover-letter.md', 'submission-information.json']:
            archive.write(ROOT/'paper'/name, name)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Overleaf archive integrity failure')
        for name, path in files.items():
            if archive.read(name) != path.read_bytes():
                raise ValueError('Packaged source differs ' + name)
    print(f'Created and verified {output}')


if __name__ == '__main__':
    main()
