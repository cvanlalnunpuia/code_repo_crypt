from __future__ import annotations

import json
from pathlib import Path

from import_linguistic_review import build_import


def main():
    root = Path(__file__).resolve().parents[1]
    outputs = build_import(root)
    for relative, expected in outputs.items():
        actual = json.loads((root / relative).read_text(encoding='utf-8'))
        if actual != expected:
            raise SystemExit(f'Imported review differs from source decisions in {relative}')
    print(f'Validated {len(outputs)} imported review resources and provenance records')


if __name__ == '__main__':
    main()
