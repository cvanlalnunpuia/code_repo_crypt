from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'archives/before-linguistic-clarification-20261007'


def main():
    ARCHIVE.mkdir(parents=True, exist_ok=False)
    hashes = {}
    names = ['configs', 'scripts', 'tests', 'resources', 'results/corpus', 'docs', 'paper', 'figures', 'output', 'corpus/annotations']
    with zipfile.ZipFile(ARCHIVE / 'baseline.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            for path in sorted((ROOT / name).rglob('*')):
                if not path.is_file() or '__pycache__' in path.parts:
                    continue
                relative = path.relative_to(ROOT).as_posix()
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                archive.write(path, relative)
        archive.writestr('baseline-manifest.json', json.dumps(hashes, sort_keys=True, indent=2) + '\n')
    with zipfile.ZipFile(ARCHIVE / 'baseline.zip') as archive:
        if archive.testzip() is not None:
            raise ValueError('Baseline archive failed integrity check')
    (ARCHIVE / 'manifest.json').write_text(json.dumps(hashes, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(f'Preserved {len(hashes)} baseline files')


if __name__ == '__main__':
    main()
