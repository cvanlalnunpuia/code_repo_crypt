from __future__ import annotations

import json
import re

from build_manuscript import OUTPUT, ROOT, build
from springer_template import OFFICIAL, preamble, bibliography, journal_settings


def validate():
    tex, supplement, ledger = build()
    if (OUTPUT / 'manuscript.tex').read_text(encoding='utf-8') != tex:
        raise ValueError('Manuscript differs from numerical source rebuild')
    if (OUTPUT / 'supplement.tex').read_text(encoding='utf-8') != supplement:
        raise ValueError('Supplement differs from source rebuild')
    stored = json.loads((OUTPUT / 'numerical-provenance.json').read_text(encoding='utf-8'))
    if stored != ledger:
        raise ValueError('Numerical ledger differs from source rebuild')
    prose = (ROOT / 'scripts/manuscript_template.tex').read_text(encoding='utf-8').split(r'\begin{abstract}', 1)[1]
    prose = re.sub(r'\\\[.*?\\\]', '', prose, flags=re.S)
    checks = {'no_em_dash': '\u2014' not in tex,
              'no_en_dash_punctuation': '\u2013' not in tex,
              'no_latex_em_dash': '---' not in tex,
              'no_semicolon_in_prose': ';' not in prose,
              'no_rhetorical_question': '?' not in prose,
              'no_explanatory_colon': ':' not in re.sub(r'\\(?:ref|cite|url)\{[^}]+\}', '', prose),
              'author_details': all(author['name'] in tex and author['email'] in tex for author in json.loads((ROOT / 'paper/submission-information.json').read_text(encoding='utf-8'))['authors']) and 'Author 1' not in tex,
              'resolved_content': '@@' not in tex and '@@' not in supplement,
              'figure_referenced': r'Figure~\ref{fig:recovery}' in tex}
    bad_words = ['notably', 'importantly', 'remarkable', 'compelling', 'delve into', 'leverage', 'showcase', 'pave the way', 'it is worth noting', 'it is important to note']
    checks['plain_prose'] = not any(word in prose.lower() for word in bad_words)
    labels = re.findall(r'\\label\{(tab:[^}]+)\}', tex)
    first_mentions = []
    for label in re.findall(r'\\ref\{(tab:[^}]+)\}', tex):
        if label not in first_mentions:
            first_mentions.append(label)
    checks['tables_referenced_in_order'] = first_mentions == labels
    figure_labels = re.findall(r'\\label\{(fig:[^}]+)\}', tex)
    figure_mentions = list(dict.fromkeys(re.findall(r'\\ref\{(fig:[^}]+)\}', tex)))
    checks['figures_referenced_in_order'] = figure_mentions == figure_labels
    checks['private_correspondence_excluded'] = 'results/corpus/linguistic-clarification.json' not in ledger['sources']
    supplement_labels = re.findall(r'\\label\{(tab:supp:[^}]+)\}', supplement)
    supplement_mentions = list(dict.fromkeys(re.findall(r'\\ref\{(tab:supp:[^}]+)\}', supplement)))
    checks['supplement_tables_referenced_in_order'] = supplement_mentions == supplement_labels
    checks['clarified_policy_described'] = 'Corpus occurrence was excluded from rule eligibility.' in tex
    checks['stale_attestation_description_absent'] = 'word-form attestation constrained' not in tex and 'word-form list built' not in tex
    scope = 'Stemming agreement was measured on the reviewed development sample.'
    checks['linguistic_agreement_scope'] = tex.count(scope) == 1 and supplement.count(scope) == 1
    author_block = tex.split(r'\begin{document}', 1)[1].split(r'\abstract{', 1)[0]
    author_positions = [author_block.index(author['name']) for author in ledger['authors']]
    checks['author_order'] = author_positions == sorted(author_positions)
    corresponding = [author for author in ledger['authors'] if author.get('corresponding')]
    checks['corresponding_author'] = len(corresponding) == 1 and corresponding[0]['email'] == ledger['corresponding_author_email'] and ('\\author*[1]{\\sur{'+corresponding[0]['name']+'}}\\email{'+ledger['corresponding_author_email']+'}') in author_block
    abstract = tex.split(r'\abstract{', 1)[1].split('\n\n\\keywords{', 1)[0].rstrip('}\n')
    for name, fact in ledger['facts'].items():
        abstract = abstract.replace('\\'+name+'{}', fact['rendered'])
    abstract = re.sub(r'\\[A-Za-z]+|[{}\\]', '', abstract)
    abstract_words = len(abstract.split())
    checks['journal_abstract_limit'] = abstract_words < 250
    checks['official_template_preamble'] = tex.startswith(preamble()) and supplement.startswith(preamble())
    checks['journal_font_settings'] = journal_settings() in tex and journal_settings() in supplement
    checks['confirmed_declarations'] = all(value in tex for field,value in json.loads((ROOT/'paper/submission-information.json').read_text(encoding='utf-8')).items() if field in ['funding_statement','competing_interests_statement','ethics_statement','author_contributions_statement'] and value)
    checks['standalone_article'] = 'Supplementary Table' not in tex and 'Online Resource' not in tex
    checks['essential_control_tables'] = all(r'\label{tab:'+name+'}' in tex for name in ['preprocessing', 'auxiliary', 'source-control', 'candidate-control', 'rules'])
    checks['official_class_unchanged'] = (ROOT/'paper/sn-jnl.cls').read_bytes() == (OFFICIAL/'sn-jnl.cls').read_bytes()
    checks['official_bibliography_style_unchanged'] = (ROOT/'paper/sn-mathphys-num.bst').read_bytes() == (OFFICIAL/'bst/sn-mathphys-num.bst').read_bytes()
    checks['external_figures_present'] = all((OUTPUT/name).read_bytes().startswith(b'%PDF-') for name in ['Fig1.pdf', 'Fig2.pdf'])
    checks['breakable_reference_urls'] = r'\usepackage{xurl}' in tex
    checks['custom_layout_overrides_absent'] = not any(command in tex for command in [r'\usepackage{geometry}', r'\familydefault', r'\captionsetup', r'\setlength{\tabcolsep}', r'\usepackage{helvet}'])
    checks['template_section_order'] = re.findall(r'\\section\{([^}]+)\}', tex) == ['Introduction', 'Results', 'Methods', 'Discussion', 'Conclusion']
    checks['portrait_main_tables'] = r'\begin{sidewaystable}' not in tex
    checks['review_process_wording_absent'] = not any(term in (tex + supplement).lower()
        for term in ['post-review', 'reviewer subsequently', 'reviewer clarification'])
    checks['probability_resolution_recomputed'] = (
        ledger['facts']['SignFlipMinimum']['value'] == 2 / 2**ledger['facts']['SeedCount']['value'] and
        ledger['facts']['LanguageHolmFloor']['value'] == ledger['facts']['LanguageFamily']['value'] * ledger['facts']['SignFlipMinimum']['value'])
    checks['descriptive_partition_scope'] = 'Statistical significance was not tested in those comparisons.' in tex
    stored_bib = (OUTPUT/'sn-bibliography.bib').read_text(encoding='utf-8')
    checks['bibliography_reconstruction'] = stored_bib == bibliography(ledger['references'])
    references = re.findall(r'@misc\{([^,]+),', stored_bib)
    citations = list(dict.fromkeys(key for group in re.findall(r'\\cite\{([^}]+)\}', tex) for key in group.split(',')))
    checks['references_in_citation_order'] = references == citations
    checks['requested_reference_minimum'] = len(references) >= 15
    commit_definition = tex.split(r'\newcommand{\CommitID}{', 1)[1].split('\n', 1)[0]
    displayed_commit = commit_definition.replace(r'\texttt{', '').replace(r'\allowbreak{}', '').rstrip('}\r')
    checks['complete_breakable_commit'] = displayed_commit == ledger['facts']['UpstreamRevision']['rendered'] and r'\allowbreak{}' in commit_definition
    checks['figure_captions_without_final_punctuation'] = all(not text.rstrip().endswith(('.', ':', ';')) for text in re.findall(r'\\caption\{([^}]+)\}', '\n'.join(re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', tex, re.S))))
    if not all(checks.values()):
        raise ValueError(f'Manuscript checks failed {checks}')
    import hashlib
    compilation = json.loads((OUTPUT/'compilation-status.json').read_text(encoding='utf-8'))
    compiled = compilation.get('state') == 'verified' and all(
        compilation.get('source_hashes', {}).get(name) == hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['paper/manuscript.tex','paper/supplement.tex'])
    result = {'checks': checks, 'abstract_word_count': abstract_words, 'reference_count': len(references), 'recomputed_fact_count': len(ledger['facts']),
              'source_count': len(ledger['sources']), 'table_row_counts': {key: len(rows) for key, rows in ledger['table_rows'].items()},
              'native_latex_compilation': 'verified with the installed pdfLaTeX compiler' if compiled else 'Final PDF inspection remains unverified for the current sources',
              'interpretation': 'Numerical and mechanical checks passed. Human scientific review and journal-specific author declarations remain required.'}
    (OUTPUT / 'validation.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2, sort_keys=True))
