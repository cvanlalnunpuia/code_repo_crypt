from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/corpus/expanded-recovery'


def main():
    subprocess.run([sys.executable, '-u', 'scripts/expanded_recovery.py'], cwd=ROOT, check=True)
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], cwd=ROOT)
    if test.returncode:
        raise RuntimeError('Expanded revision test suite failed')
    subprocess.run([sys.executable, 'scripts/build_manuscript.py'], cwd=ROOT, check=True)
    ledger = json.loads((ROOT/'paper/numerical-provenance.json').read_text(encoding='utf-8'))
    if 'results/corpus/expanded-recovery/expanded-analysis.json' not in ledger['sources']:
        raise RuntimeError('Expanded results have not been incorporated into the manuscript builder')
    for script in ['validate_manuscript.py', 'compile_project_pdfs.py', 'check_compiled_pdf_bounds.py']:
        subprocess.run([sys.executable, 'scripts/'+script], cwd=ROOT, check=True)
    (OUT/'completion-status.json').write_text(json.dumps(dict(stage='sources_and_pdfs_ready',
        final_visual_inspection='pending', native_editor_compilation='pending',
        packages='Refresh after final visual inspection'), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
