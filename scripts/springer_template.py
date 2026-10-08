from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / 'paper/official-template/package/sn-article-template'


def preamble():
    original = (OFFICIAL / 'sn-article.tex').read_text(encoding='utf-8', errors='replace')
    return original.split(r'\begin{document}', 1)[0]


def frontmatter(authors):
    affiliations = list(dict.fromkeys(a['affiliation'] for a in authors))
    lines = []
    for author in authors:
        number = affiliations.index(author['affiliation']) + 1
        star = '*' if author.get('corresponding') else ''
        lines.append(f"\\author{star}[{number}]{{\\sur{{{author['name']}}}}}\\email{{{author['email']}}}")
    for index, affiliation in enumerate(affiliations, 1):
        department, institution, city, postcode = affiliation.split(', ')
        lines.append(f'\\affil[{index}]{{\\orgdiv{{{department}}}, \\orgname{{{institution}}}, \\orgaddress{{\\city{{{city}}}, \\postcode{{{postcode}}}}}}}')
    return '\n\n'.join(lines)


def figure_block(legacy, label, filename):
    block = next(block for block in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', legacy, re.S) if '\\label{'+label+'}' in block)
    caption = block[block.index(r'\caption{'):block.index(r'\label{')]
    return '\n'.join([r'\begin{figure}[htbp]', r'\centering', f'\\includegraphics[width=\\textwidth]{{{filename}}}', caption.strip(), '\\label{'+label+'}', r'\end{figure}'])


def migrate(legacy, supplement, authors):
    abstract = legacy.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0].strip()
    facts = '\n'.join(line for line in legacy.splitlines() if line.startswith(r'\newcommand'))
    title = re.search(r'\\title\{([^}]+)\}', legacy).group(1)
    body = legacy[legacy.index(r'\section{Introduction}'):legacy.index(r'\section*{Data availability}')]
    discussion_start, methods_start = body.index(r'\section{Discussion}'), body.index(r'\section{Methods}')
    body = body[:discussion_start] + body[methods_start:] + body[discussion_start:methods_start]
    for label, filename in [('fig:recovery', 'Fig1.pdf'), ('fig:partition', 'Fig2.pdf')]:
        replacement = figure_block(legacy, label, filename)
        body = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', lambda m:replacement if '\\label{'+label+'}' in m.group() else m.group(), body, flags=re.S)
    body = body.replace(r'\bottomrule', r'\botrule')
    body = re.sub(r'\\begin\{table\}.*?\\end\{table\}', lambda m:m.group().replace(r'\begin{table}', r'\begin{sidewaystable}').replace(r'\end{table}', r'\end{sidewaystable}') if r'\label{tab:ablation}' in m.group() else m.group(), body, flags=re.S)
    availability = legacy.split(r'\section*{Data availability}', 1)[1].split(r'\begin{thebibliography}', 1)[0].strip()
    main = (preamble() + '\n\\usepackage{xurl}\n' + facts + '\n\\begin{document}\n' +
            f'\\title[English-Mizo IHOP Query Recovery]{{{title}}}\n\n' + frontmatter(authors) + '\n\n\\abstract{'+abstract+'}\n\n'+
            r'\keywords{searchable symmetric encryption, query recovery, Mizo, preprocessing, corpus evaluation}' +
            '\n\n\\maketitle\n\n' + body + '\n\\backmatter\n\n\\bmhead{Supplementary information}\n\n'+
            'Supplementary material contains full recovery comparisons, reviewed linguistic resources, and numerical source identifiers.\n\n'+
            '\\section*{Declarations}\n\\subsection*{Data availability}\n'+availability+'\n\n\\bibliography{sn-bibliography}\n\\end{document}\n')
    supplemental_body = supplement.split(r'\begin{document}', 1)[1].split(r'\end{document}', 1)[0]
    supplemental_body = supplemental_body.replace(r'\bottomrule', r'\botrule')
    supplemental = (preamble() + '\n\\usepackage{xurl,longtable,pdflscape}\n' + facts + '\n\\begin{document}\n'+
                    '\\title[Supplementary material]{Supplementary material for '+title+'}\n\n'+frontmatter(authors)+
                    '\n\\maketitle\n\\begin{landscape}\n'+supplemental_body+'\n\\end{landscape}\n\\end{document}\n')
    return main, supplemental


def bibliography(references):
    entries = []
    for reference in references:
        key, names, title, venue, url = [reference[field] for field in ['key','authors','title','venue','url']]
        if names in ['NLLB Team', 'Meta', 'Snowball project', 'Carnegie Mellon University']:
            names = '{'+names+'}'
        else:
            names = names.replace(', and ', ', ').replace(' and ', ', ')
            names = ' and '.join(names.split(', '))
        year = re.search(r'\b(?:19|20)\d{2}\b', venue)
        # The supplied bibliography style converts literal tildes into text commands.
        # Percent encoding preserves the URL destination inside its URL command.
        fields = dict(author=names, title='{'+title+'}', howpublished=venue, url=url.replace('~', '%7E'))
        if year:
            fields['year'] = year.group()
        if url.startswith('https://doi.org/'):
            fields['doi'] = url.removeprefix('https://doi.org/')
            del fields['url']
        entries.append('@misc{'+key+',\n'+',\n'.join('  '+field+' = {'+value+'}' for field,value in fields.items())+'\n}')
    return '\n\n'.join(entries)+'\n'
