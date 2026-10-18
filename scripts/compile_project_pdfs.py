from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    compiler = shutil.which('pdflatex')
    bibliography = shutil.which('bibtex')
    if not compiler or not bibliography:
        raise RuntimeError('The installed TeX compiler and bibliography processor are required')
    output = ROOT/'output/pdf'
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for key in ['TEXINPUTS', 'BIBINPUTS', 'BSTINPUTS']:
        environment[key] = str(ROOT/'paper')+os.pathsep
    commands = []
    for name in ['manuscript', 'supplement']:
        command = [compiler, '-interaction=nonstopmode', '-halt-on-error',
                   '-output-directory='+str(output), name+'.tex']
        invocations = [command]
        if name == 'manuscript':
            invocations.append([bibliography, name])
        invocations.extend([command, command])
        for invocation in invocations:
            working_directory = output if invocation[0] == bibliography else ROOT/'paper'
            result = subprocess.run(invocation, cwd=working_directory, env=environment,
                                    text=True, capture_output=True)
            commands.append(dict(command=invocation, exit_code=result.returncode))
            if result.returncode:
                raise RuntimeError(result.stdout[-4000:]+result.stderr[-1000:])
        path = output/(name+'.pdf')
        if not path.read_bytes().startswith(b'%PDF-'):
            raise ValueError('Expected a compiled PDF ' + name)
    record = dict(state='compiled', visual_inspection='pending', commands=commands,
        source_hashes={str(path.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [ROOT/'paper/manuscript.tex', ROOT/'paper/supplement.tex']},
        pdf_hashes={name:hashlib.sha256((output/(name+'.pdf')).read_bytes()).hexdigest() for name in ['manuscript','supplement']})
    (ROOT/'paper/compilation-status.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(dict(state=record['state'], visual_inspection=record['visual_inspection'], pdf_hashes=record['pdf_hashes']), indent=2))


if __name__ == '__main__':
    main()
