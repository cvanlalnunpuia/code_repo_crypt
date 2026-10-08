from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pdfplumber
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]


def validate():
    ledger = json.loads((ROOT / 'paper/numerical-provenance.json').read_text(encoding='utf-8'))
    results = {}
    texts = {}
    glyphs = TTFont('ValidationArial', 'C:/Windows/Fonts/arial.ttf').face.charToGlyph
    for name in ['manuscript', 'supplement']:
        path = ROOT / 'output/pdf' / (name + '.pdf')
        with pdfplumber.open(path) as document:
            pages = []
            for number, page in enumerate(document.pages, 1):
                text = page.extract_text() or ''
                if len(text.strip()) < 30:
                    raise ValueError(f'Empty page {name} {number}')
                for word in page.extract_words():
                    if word['x0'] < 45 or word['x1'] > page.width - 45 or word['top'] < 35 or word['bottom'] > page.height - 20:
                        raise ValueError(f'Text outside margins {name} {number} {word}')
                if '\ufffd' in text or '@@' in text:
                    raise ValueError(f'Unresolved content {name} {number}')
                unsupported = {character for character in text if not character.isspace() and ord(character) not in glyphs}
                if unsupported:
                    raise ValueError(f'Unsupported font glyphs {name} {number} {unsupported}')
                pages.append(text)
            texts[name] = '\n'.join(pages)
            results[name] = {'page_count': len(pages), 'text_within_margins': True,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    main = texts['manuscript']
    for name, fact in ledger['facts'].items():
        if fact['rendered'] not in main:
            raise ValueError('Missing generated fact ' + name)
    for rows in ledger['table_rows'].values():
        for row in rows:
            for cell in row:
                for value in re.findall(r'-?\d+\.\d+', str(cell)):
                    if value not in main:
                        raise ValueError('Missing table value ' + value)
    for checksum in ledger['sources'].values():
        if checksum not in texts['supplement'].replace('\n', ''):
            raise ValueError('Missing source checksum ' + checksum)
    results['checks'] = {'generated_facts_present': True, 'table_values_present': True,
                         'source_identifiers_present': True, 'font_glyphs_supported': True}
    (ROOT / 'paper/pdf-validation.json').write_text(json.dumps(results, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return results


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2, sort_keys=True))
