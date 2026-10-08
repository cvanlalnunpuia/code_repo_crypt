from __future__ import annotations

import html
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/pdf'
TITLE = 'Corpus and Preprocessing Effects on IHOP Query Recovery in English and Mizo'
WIDTH = A4[0] - 100
for name, filename in [('Paper', 'arial.ttf'), ('PaperBold', 'arialbd.ttf'), ('PaperItalic', 'ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/' + filename))
pdfmetrics.registerFontFamily('Paper', normal='Paper', bold='PaperBold', italic='PaperItalic', boldItalic='PaperBold')
STYLES = getSampleStyleSheet()
BODY = ParagraphStyle('PaperBody', fontName='Paper', fontSize=10, leading=14, spaceAfter=7)
CELL = ParagraphStyle('PaperCell', fontName='Paper', fontSize=7.6, leading=10)
CAPTION = ParagraphStyle('PaperCaption', parent=BODY, fontSize=8.6, leading=11, spaceAfter=5)
HEADING = ParagraphStyle('PaperHeading', parent=BODY, fontName='PaperBold', fontSize=13, spaceBefore=12, spaceAfter=7, keepWithNext=True)
SUBHEADING = ParagraphStyle('PaperSubheading', parent=HEADING, fontSize=11, spaceBefore=8)
TITLE_STYLE = ParagraphStyle('PaperTitle', parent=HEADING, fontSize=19, leading=23, alignment=TA_CENTER, spaceAfter=12)
CENTRED = ParagraphStyle('PaperCentred', parent=BODY, alignment=TA_CENTER, fontSize=9, leading=12)


def clean(text, ledger):
    facts = ledger['facts']
    text = re.sub(r'\\([A-Za-z]+)\{\}', lambda m: facts[m[1]]['rendered'] if m[1] in facts else m[0], text)
    keys = {x['key']: i for i, x in enumerate(ledger['references'], 1)}
    text = re.sub(r'\\cite\{([^}]+)\}', lambda m: '[' + ', '.join(str(keys[x]) for x in m[1].split(',')) + ']', text)
    labels = {'tab:recovery': 1, 'tab:language': 2, 'tab:structure': 3, 'tab:balanced': 4, 'tab:ablation': 5, 'tab:corpora': 6, 'fig:recovery': 1}
    text = re.sub(r'\\ref\{([^}]+)\}', lambda m: str(labels[m[1]]), text)
    text = re.sub(r'\\(?:texttt|textbf)\{([^}]+)\}', r'\1', text)
    text = text.replace(r'\%', '%').replace(r'\&', '&').replace(r'\_', '_').replace('~', ' ')
    text = text.replace(r'\noindent', '').replace(r'\quad', ' ').replace('$', '')
    text = text.replace(r'\hat q_t', 'qhat_t')
    text = ' '.join(text.split())
    if '\\' in text:
        raise ValueError('Unconverted LaTeX in paragraph ' + text)
    return text


def paragraph(text, ledger, style=BODY):
    content = html.escape(clean(text, ledger))
    content = content.replace('qhat_t', '<i>q̂</i><sub>t</sub>').replace('q_t', '<i>q</i><sub>t</sub>')
    return Paragraph(content, style)


def pdf_table(caption, headers, rows, widths=None):
    converted = [[Paragraph(html.escape(str(cell).replace(r'\%', '%').replace(r'\&', '&').replace('$', '')), CELL) for cell in row] for row in [headers] + rows]
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                               ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                               ('LINEABOVE', (0, 0), (-1, 0), 0.7, colors.black), ('LINEBELOW', (0, 0), (-1, 0), 0.4, colors.black),
                               ('LINEBELOW', (0, -1), (-1, -1), 0.7, colors.black)]))
    return [Paragraph(html.escape(caption), CAPTION), table, Spacer(1, 10)]


def page(canvas, document):
    canvas.setAuthor('')
    canvas.setCreator('')
    canvas._doc.info.producer = ''
    canvas.setTitle(TITLE if 'manuscript' in document.filename else 'Supplementary material')
    canvas.setFont('Paper', 8)
    canvas.drawRightString(A4[0] - 50, 28, str(document.page))


def save(path, story):
    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=50, leftMargin=50, topMargin=45, bottomMargin=45,
                      title=TITLE, author='', pageCompression=1).build(story, onFirstPage=page, onLaterPages=page)


def build_main(ledger):
    template = (ROOT / 'scripts/manuscript_template.tex').read_text(encoding='utf-8')
    text = template.split(r'\maketitle', 1)[1].split(r'\end{document}', 1)[0]
    text = text.replace(r'\begin{abstract}', r'\section{Abstract}').replace(r'\end{abstract}', '')
    parts = re.split(r'(\\(?:sub)?section\{[^}]+\}|@@[A-Z_]+@@|\\\[.*?\\\])', text, flags=re.S)
    story = [Paragraph(TITLE, TITLE_STYLE), Paragraph('Author 1, Author 2, Author 3, Author 4, Author 5', CENTRED),
             Paragraph('Affiliation 1<br/>Affiliation 2<br/>Affiliation 3<br/>Affiliation 4<br/>Affiliation 5', CENTRED), Spacer(1, 7)]
    captions = {
        'RECOVERY_TABLE': (1, 'recovery', ['Language','Condition','Queries','Weighted (SD)','Macro (SD)'], [60,75,52,145,WIDTH-332], 'Final recovery means and sample standard deviations.'),
        'LANGUAGE_TABLE': (2, 'language', ['Condition','Queries','Metric','Difference','95% CI','Holm p'], [72,52,55,65,145,WIDTH-389], 'Mizo minus English recovery at the final checkpoint.'),
        'STRUCTURE_TABLE': (3, 'structure', ['Language','Condition','Density','Alternatives','Nearest Jaccard','Cosine'], [62,82,60,82,112,WIDTH-398], 'Client-index density, frequency alternatives, and median similarities.'),
        'BALANCED_TABLE': (4, 'balanced_selected', ['Language','K','FLORES','WMT23','Difference','95% CI','Holm p'], [55,30,58,58,60,140,WIDTH-401], 'Selected Zipf query-weighted recovery contrasts after length balancing.'),
        'ABLATION_TABLE': (5, 'enron_endpoints', ['Intervention','Endpoint contrast','Difference','95% CI'], [110,180,65,WIDTH-355], 'Paired final-recovery differences at the configured Enron intervention endpoints.'),
        'CORPUS_TABLE': (6, 'corpora', ['Collection','Pairs','Auxiliary','Client'], [WIDTH-195,65,65,65], 'Paired collection sizes and fixed partition sizes per language.'),
    }
    for part in parts:
        if not part.strip():
            continue
        if part.startswith(r'\section') or part.startswith(r'\subsection'):
            heading = re.search(r'\{([^}]+)\}', part)[1]
            story.append(Paragraph(heading, SUBHEADING if part.startswith(r'\subsection') else HEADING))
        elif part.startswith('@@'):
            key = part.strip('@')
            if key in captions:
                number, source, headers, widths, caption = captions[key]
                story.extend(pdf_table(f'Table {number}. {caption}', headers, ledger['table_rows'][source], widths))
            elif key == 'RECOVERY_FIGURE':
                story.append(KeepTogether([Image(str(ROOT / 'figures/production-recovery.png'), width=WIDTH, height=WIDTH * 2 / 3),
                                          Paragraph('Figure 1. Mean query-weighted recovery by iteration. Colours identify preprocessing. Panels show English and Mizo under uniform and Zipf queries.', CAPTION)]))
            elif key == 'REFERENCES':
                story.append(Paragraph('References', HEADING))
                for i, r in enumerate(ledger['references'], 1):
                    body = html.escape(f"[{i}] {r['authors']}. {r['title']}. {r['venue']}.")
                    story.append(Paragraph(body + ' <link href="' + html.escape(r['url'], quote=True) + '">Source</link>.', CAPTION))
            else:
                raise ValueError('Unknown content marker ' + key)
        elif part.startswith(r'\['):
            if r'R_{\mathrm{weighted}}' in part:
                expression = 'R<sub>weighted</sub> = (1/T) Σ<sub>t=1</sub><super>T</super> 1(q̂<sub>t</sub> = q<sub>t</sub>)'
            else:
                expression = 'R<sub>macro</sub> = (1/|U|) Σ<sub>u in U</sub> [Σ<sub>t:q_t=u</sub> 1(q̂<sub>t</sub> = u)] / |{t : q<sub>t</sub> = u}|'
            preceding = story.pop()
            story.append(KeepTogether([preceding, Paragraph(expression, CENTRED)]))
        else:
            for paragraph_text in re.split(r'\n\s*\n', part):
                if paragraph_text.strip():
                    story.append(paragraph(paragraph_text, ledger))
    save(OUTPUT / 'manuscript.pdf', story)


def build_supplement(ledger):
    source = (ROOT / 'paper/supplement.tex').read_text(encoding='utf-8')
    tables = re.findall(r'\\begin\{longtable\}.*?\\end\{longtable\}', source, flags=re.S)
    story = [Paragraph('Supplementary material', TITLE_STYLE),
             Paragraph('Full recovery comparisons and development-sample stemming disagreements. Differences were paired across seeds. Recovery measurements were proportions.', BODY)]
    for i, text in enumerate(tables, 1):
        caption = re.search(r'\\caption\{([^}]+)\}', text)[1]
        header = text.split(r'\toprule', 1)[1].split(r'\\', 1)[0]
        headers = [clean(cell.strip(), ledger) for cell in header.split('&')]
        raw = text.split(r'\endhead', 1)[1].split(r'\bottomrule', 1)[0]
        rows = [[clean(cell.strip(), ledger) for cell in row.split('&')] for row in raw.split(r'\\') if row.strip()]
        count = len(headers)
        widths = [WIDTH / count] * count
        if i == 1:
            widths = [50,60,60,46,48,48,126,WIDTH-438]
        elif count == 8:
            widths = [53,32,50,50,65,65,65,WIDTH-380]
        story.extend(pdf_table(f'Table S{i}. {caption}', headers, rows, widths))
    story.append(Paragraph('The English and Mizo annotation agreement measurements were development-sample results. Independent held-out accuracy was not estimated.', BODY))
    story.append(Paragraph('Provenance', HEADING))
    story.append(Paragraph('SHA-256 identifiers for the source results and configurations were recorded below. Source text was excluded.', BODY))
    for name, checksum in ledger['sources'].items():
        story.append(Paragraph(html.escape(name) + '<br/>' + checksum, CAPTION))
    save(OUTPUT / 'supplement.pdf', story)


def main():
    ledger = json.loads((ROOT / 'paper/numerical-provenance.json').read_text(encoding='utf-8'))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_main(ledger)
    build_supplement(ledger)
    print('Rendered manuscript and supplementary PDFs from the verified manuscript content')


if __name__ == '__main__':
    main()
