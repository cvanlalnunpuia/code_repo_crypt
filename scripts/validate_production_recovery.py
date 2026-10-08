from __future__ import annotations

import json
from pathlib import Path

from analyse_production_recovery import build, markdown


def main():
    root = Path(__file__).resolve().parents[1]
    rebuilt = build(root)
    stored = json.loads((root / 'results/corpus/production-recovery-analysis.json').read_text(encoding='utf-8'))
    if stored != rebuilt:
        raise SystemExit('Production analysis differs from independent reconstruction')
    if (root / 'docs/production-recovery.md').read_text(encoding='utf-8') != markdown(rebuilt):
        raise SystemExit('Production report differs from independent reconstruction')
    print(f"Validated all reported values from {rebuilt['run_count']} production runs")


if __name__ == '__main__':
    main()
