from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12'


def main():
    expected = json.loads((ROOT/'template-dependencies.json').read_text(encoding='utf-8'))
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name, checksum in expected.items():
            if name == 'paper/sn-mathphys-num.bst':
                source = 'bst/sn-mathphys-num.bst'
            elif name == 'paper/sn-jnl.cls':
                source = 'sn-jnl.cls'
            else:
                source = name.split('sn-article-template/', 1)[1]
            matches = [member for member in archive.namelist() if member.endswith('/'+source)]
            if len(matches) != 1:
                raise ValueError('Template asset was missing or ambiguous '+source)
            content = archive.read(matches[0])
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError('Publisher template version changed '+source)
            target = ROOT/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print('Official template dependencies obtained and hashes verified')


if __name__ == '__main__':
    main()
