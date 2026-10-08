from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '10631a10e3ec461ad01d38fc74f7933d31062ba5'


def main():
    directory = ROOT/'vendor/ihop-code'
    directory.parent.mkdir(exist_ok=True)
    if not directory.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--sparse', '--no-checkout',
                        'https://github.com/simon-oya/ihop-code.git', str(directory)], check=True)
        subprocess.run(['git', '-C', str(directory), 'sparse-checkout', 'set', 'attacks', 'processing'], check=True)
        subprocess.run(['git', '-C', str(directory), 'checkout', REVISION], check=True)
    actual = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise ValueError('Existing upstream checkout has a different revision')
    (ROOT/'vendor/ihop-code.commit').write_text(REVISION+'\n', encoding='utf-8')
    print('Pinned upstream source verified. Dataset directories were excluded from the sparse checkout.')


if __name__ == '__main__':
    main()
