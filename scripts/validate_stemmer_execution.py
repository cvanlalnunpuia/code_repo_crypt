from __future__ import annotations

import json
from pathlib import Path

from analyse_stemmer_execution import build, markdown


def main():
    root = Path(__file__).resolve().parents[1]
    result = build(root)
    stored = json.loads((root / 'results/corpus/stemmer-execution-evaluation.json').read_text(encoding='utf-8'))
    if result != stored:
        raise SystemExit('Stored stemmer evaluation differs from reconstruction')
    if markdown(result) != (root / 'docs/stemmer-execution.md').read_text(encoding='utf-8'):
        raise SystemExit('Stemmer report differs from reconstruction')
    print(f"Validated {result['overall']['item_count']} fixed annotations, clarified examples and auxiliary comparison types")


if __name__ == '__main__':
    main()
