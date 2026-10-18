from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / 'paper/official-template/package/sn-article-template'


def preamble():
    original = (OFFICIAL / 'sn-article.tex').read_text(encoding='utf-8', errors='replace')
    return original.split(r'\begin{document}', 1)[0]


def journal_settings():
    return r'''
\usepackage{array,tabularx,ragged2e}
\hyphenation{To-ken-isa-tion Stop-words Fre-quen-cy Al-ter-na-tives}
\makeatletter
\renewcommand\normalsize{\@setfontsize\normalsize{12pt}{14.4pt}}
\renewcommand\small{\normalsize}
\renewcommand\footnotesize{\normalsize}
\def\figurecaptionfont{\normalfont\normalsize}
\def\tablecaptionfont{\normalfont\normalsize}
\def\addressfont{\normalfont\normalsize\titraggedcenter}
\def\abstractheadfont{\normalfont\normalsize\bfseries\titraggedcenter}
\def\abstractsubheadfont{\normalfont\normalsize\bfseries}
\def\abstractfont{\normalfont\normalsize\leftskip=24pt\rightskip=24pt\parfillskip=0pt plus 1fil}
\def\keywordfont{\normalfont\normalsize\leftskip=24pt\rightskip=24pt plus0.5fill}
\def\bmheadfont{\normalfont\normalsize\bfseries\raggedright\boldmath}
\def\tablebodyfont{\normalfont\normalsize}
\def\tablecolheadfont{\normalfont\normalsize\bfseries}
\def\tablefootnotefont{\normalfont\normalsize}
\makeatother
\AtBeginDocument{\normalsize}
'''


def wrap_tables(body):
    def replace(match):
        block = match.group()
        sideways = r'\begin{sidewaystable}' in block
        spec = re.search(r'\\begin\{tabular\}\{(@\{\}.*?@\{\})\}', block)
        if spec is None:
            return block
        header = block.split(r'\toprule', 1)[1].split(r'\midrule', 1)[0]
        count = header.count('&') + 1
        if count >= 5 and not sideways:
            block = block.replace(r'\begin{table}', r'\begin{sidewaystable}').replace(r'\end{table}', r'\end{sidewaystable}')
            sideways = True
        columns = '@{}' + r'>{\RaggedRight\arraybackslash}X' * count + '@{}'
        width = r'\textheight' if sideways else r'\linewidth'
        return (block.replace(spec.group(), r'\begin{tabularx}{'+width+'}{'+columns+'}')
                .replace(r'\end{tabular}', r'\end{tabularx}'))
    return re.sub(r'\\begin\{(?:table|sidewaystable)\}.*?\\end\{(?:table|sidewaystable)\}', replace, body, flags=re.S)


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
    return '\n'.join([block.splitlines()[0], r'\centering', f'\\includegraphics[width=\\textwidth]{{{filename}}}', caption.strip(), '\\label{'+label+'}', r'\end{figure}'])


def migrate(legacy, supplement, authors, submission=None):
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
    availability = legacy.split(r'\section*{Data availability}', 1)[1].split(r'\begin{thebibliography}', 1)[0].strip()
    body = wrap_tables(body)
    declarations = ''
    for field, heading in [('funding_statement', 'Funding'), ('competing_interests_statement', 'Competing interests'),
                           ('author_contributions_statement', 'Author contributions'), ('ethics_statement', 'Ethics approval'),
                           ('writing_support_statement', 'Writing support')]:
        value = (submission or {}).get(field)
        if value:
            declarations += '\\subsection*{'+heading+'}\n'+value+'\n\n'
    main = (preamble() + journal_settings() + '\n\\usepackage{xurl}\n' + facts + '\n\\begin{document}\n' +
            f'\\title[English-Mizo IHOP Query Recovery]{{{title}}}\n\n' + frontmatter(authors) + '\n\n\\abstract{'+abstract+'}\n\n'+
            r'\keywords{searchable symmetric encryption, query recovery, Mizo, preprocessing, corpus evaluation}' +
            '\n\n\\maketitle\n\n' + body + '\n\\backmatter\n\n\\bmhead*{Supplementary information}\n\n'+
            'Online Resource 1 (Supplementary Information.pdf) contains full recovery comparisons, reviewed linguistic resources, and numerical source identifiers. Online Resource 2 (reproducibility-release.zip) contains recorded results, configurations and numerical reconstruction scripts.\n\n'+
            '\\section*{Declarations}\n'+declarations+'\\subsection*{Data availability}\n'+availability+'\n\n\\bibliography{sn-bibliography}\n\\end{document}\n')
    supplemental_body = supplement.split(r'\begin{document}', 1)[1].split(r'\end{document}', 1)[0]
    supplemental_body = supplemental_body.replace(r'\bottomrule', r'\botrule')
    landscape_footer = r'''
\makeatletter
\def\ps@supplement{\let\@oddhead\@empty\let\@evenhead\@empty
\def\@oddfoot{\reset@font\hfil\thepage\hfil}\let\@evenfoot\@oddfoot}
\makeatother
'''
    supplemental = (preamble() + journal_settings() + '\n\\usepackage{xurl,longtable,pdflscape}\n' + landscape_footer + facts + '\n\\begin{document}\n'+
                    '\\title[Supplementary material]{Supplementary material for '+title+'}\n\n'+frontmatter(authors)+
                    '\n\\maketitle\n\\begin{landscape}\n\\pagestyle{supplement}\n'+supplemental_body+'\n\\end{landscape}\n\\end{document}\n')
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
        safe_title = title.replace('%', r'\%').replace('<', r'\textless{}')
        fields = dict(author=names, title='{'+safe_title+'}', howpublished=venue, url=url.replace('~', '%7E'))
        if year:
            fields['year'] = year.group()
        else:
            fields['note'] = 'Accessed 9 October 2026'
        if url.startswith('https://doi.org/'):
            fields['doi'] = url.removeprefix('https://doi.org/')
            del fields['url']
        entries.append('@misc{'+key+',\n'+',\n'.join('  '+field+' = {'+value+'}' for field,value in fields.items())+'\n}')
    return '\n\n'.join(entries)+'\n'
