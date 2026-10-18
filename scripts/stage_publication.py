"""Stage only the audited reconstruction files in a fresh publication checkout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = 'https://github.com/cvanlalnunpuia/code_repo_crypt.git'
DEST = ROOT/'output/github-publish-v1.1.0'


def main():
    audit = json.loads((ROOT/'output/github-release-audit.json').read_text(encoding='utf-8'))
    package = ROOT/'output/reproducibility-release.zip'
    with zipfile.ZipFile(package) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive integrity failed')
        manifest = json.loads(archive.read('release-manifest.json'))
        if manifest != audit['file_sha256']:
            raise ValueError('Archive differs from the publication audit')
        if set(archive.namelist()) != set(manifest) | {'release-manifest.json'}:
            raise ValueError('Unexpected archive member')
        for name, checksum in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise ValueError('Archive checksum failed '+name)
        if DEST.exists():
            raise ValueError('Publication checkout already exists')
        subprocess.run(['git', 'clone', REMOTE, str(DEST)], check=True)
        parent = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=DEST, text=True).strip()
        for name in manifest:
            target = DEST/name
            if not target.resolve().is_relative_to(DEST.resolve()) or '.git' in Path(name).parts:
                raise ValueError('Unsafe archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    shutil.copyfile(package, DEST/'reproducibility-release.zip')
    for field in ['user.name', 'user.email']:
        value = subprocess.check_output(['git', 'config', '--get', field], cwd=ROOT/'output/github-release', text=True).strip()
        subprocess.run(['git', 'config', '--local', field, value], cwd=DEST, check=True)
    names = sorted(manifest) + ['reproducibility-release.zip']
    for start in range(0, len(names), 40):
        subprocess.run(['git', 'add', '--'] + names[start:start+40], cwd=DEST, check=True)
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=DEST, text=True).splitlines()
    if not set(staged).issubset(set(names)):
        raise ValueError('Unaudited file staged')
    subprocess.run(['git', 'diff', '--cached', '--stat'], cwd=DEST, check=True)
    subprocess.run(['git', 'commit', '-m', 'Publish verified expanded recovery reconstruction v1.1.0'], cwd=DEST, check=True)
    record = dict(parent=parent,
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=DEST, text=True).strip(),
                  version='v1.1.0', audited_files=len(manifest),
                  archive_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
                  old_tags_preserved=True, published=False)
    (ROOT/'output/publication-stage.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
