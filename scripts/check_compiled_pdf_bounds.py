from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'tmp/pdfs'
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in ['manuscript', 'supplement']:
        pdf = ROOT/'output/pdf'/(name+'.pdf')
        bbox = output/(name+'-bounds.xhtml')
        subprocess.run(['pdftotext', '-bbox-layout', str(pdf), str(bbox)], check=True, capture_output=True)
        root = ET.parse(bbox).getroot()
        ns = {'x':'http://www.w3.org/1999/xhtml'}
        pages = root.findall('.//x:page', ns)
        if not pages:
            raise ValueError('No PDF pages were extracted')
        bounds = []
        info = subprocess.run(['pdfinfo', '-f', '1', '-l', str(len(pages)), str(pdf)], check=True, capture_output=True, text=True).stdout
        rotations = {int(number):int(angle) for number,angle in re.findall(r'Page\s+(\d+)\s+rot:\s+(\d+)', info)}
        for number, page in enumerate(pages, 1):
            width, height = float(page.attrib['width']), float(page.attrib['height'])
            if rotations.get(number, 0) % 180:
                width, height = height, width
            words = page.findall('.//x:word', ns)
            if len(words) < 5:
                raise ValueError('Unexpected near-empty page ' + name+' '+str(number))
            for word in words:
                coords = [float(word.attrib[key]) for key in ['xMin','yMin','xMax','yMax']]
                if coords[0] < 0 or coords[1] < 0 or coords[2] > width+.5 or coords[3] > height+.5:
                    bounds.append(dict(page=number, text=word.text, bounds=coords))
        result[name] = dict(page_count=len(pages), out_of_page_words=bounds,
                            sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
    if any(value['out_of_page_words'] for value in result.values()):
        raise ValueError(json.dumps({key:dict(page_count=value['page_count'], outside_count=len(value['out_of_page_words']), examples=value['out_of_page_words'][:5]) for key,value in result.items()}, indent=2))
    (ROOT/'paper/pdf-bounds-check.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
