import json

from analyse_export_structure import ROOT, build, report

result = build()
stored = json.loads((ROOT / 'results/corpus/export-structure-analysis.json').read_text(encoding='utf-8'))
if stored != result:
    raise SystemExit('Stored export structure differs from recomputation')
if (ROOT / 'docs/export-structure.md').read_text(encoding='utf-8') != report(result):
    raise SystemExit('Stored structure report differs from recomputation')
print(f"Validated {len(result['records'])} exact export profiles")
