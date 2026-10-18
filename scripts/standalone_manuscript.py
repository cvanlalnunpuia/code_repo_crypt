"""Consolidate essential evidence from the reconstructed numerical tables."""
from __future__ import annotations

import re
import json
from pathlib import Path

RELEASE_URL = 'https://github.com/cvanlalnunpuia/code_repo_crypt/releases/tag/v1.1.0'


def source_rows(supplement, number):
    block = next(block for block in re.findall(r'\\begin\{longtable\}.*?\\end\{longtable\}', supplement, re.S)
                 if r'\label{tab:supp:' + str(number) + '}' in block)
    body = block.split(r'\midrule\endhead', 1)[1].split(r'\botrule', 1)[0]
    return [row.strip().split(' & ') for row in body.split(r'\\') if row.strip()]


def table(label, caption, headers, rows, weights):
    columns = '@{}' + ''.join(r'>{\hsize=' + str(w) + r'\hsize\linewidth=\hsize\RaggedRight\arraybackslash}X' for w in weights) + '@{}'
    return '\n'.join([r'\begin{table}[!htbp]', r'\centering',
        r'\caption{' + caption + r'}\label{tab:' + label + '}',
        r'\begin{tabularx}{\linewidth}{' + columns + '}', r'\toprule',
        ' & '.join(headers) + r' \\', r'\midrule',
        *[' & '.join(row) + r' \\' for row in rows],
        r'\botrule', r'\end{tabularx}', r'\end{table}', ''])


def paired(a, b):
    return r'\shortstack[l]{{' + a + r'}\\{' + b + '}}'


def consolidate(tex, supplement):
    preamble, tex = tex.split(r'\begin{document}', 1)
    rows = source_rows(supplement, 1)
    contrasts = []
    for first, second in zip(rows[::2], rows[1::2]):
        assert first[:4] == second[:4] and first[4] == 'Weighted' and second[4] == 'Macro'
        label = first[1] + ' minus ' + first[2] + r'\newline ' + first[3]
        contrasts.append([label, paired(first[5], second[5]), paired(first[6], second[6]), paired(first[7], second[7])])
    caption = ' preprocessing contrasts. Weighted estimates precede macro estimates within each cell. Confidence intervals were unadjusted.'
    block = table('preprocessing', 'English' + caption,
                  ['Contrast and queries', r'Differ\-ence', '95\\% CI', 'Holm $p$'], contrasts[:6], [1.05, .6, 1.7, .65])
    block += table('preprocessing-mizo', 'Mizo' + caption,
                   ['Contrast and queries', r'Differ\-ence', '95\\% CI', 'Holm $p$'], contrasts[6:], [1.05, .6, 1.7, .65])
    tex = tex.replace(r'Supplementary Table S1 reports the preprocessing contrasts,', r'Tables~\ref{tab:preprocessing} and \ref{tab:preprocessing-mizo} report the preprocessing contrasts,')
    # The language table was mentioned first and remains first in the source.
    location = tex.index(r'Figure~\ref{fig:partition} shows selected paired contrasts')
    tex = tex[:location] + block + '\n' + tex[location:]
    # Move the preprocessing table's first mention after the language table.
    sentence = r'Tables~\ref{tab:preprocessing} and \ref{tab:preprocessing-mizo} report the preprocessing contrasts, of which \PreprocessingSignificant{} remained below the corrected threshold. The stemmed versus stopword-only contrasts remained above it.'
    tex = tex.replace(sentence, '')
    tex = tex.replace(block, sentence + '\n\n' + block)

    auxiliary = source_rows(supplement, 18)
    aux_rows = [[row[1] + r'\newline ' + row[2], row[3], row[4], row[5]] for row in auxiliary if row[0] == 'auxiliary only']
    shift_rows = [[row[1] + r'\newline ' + {'flores200':'FLORES-200', 'wmt23':'WMT23'}[row[2]], row[3], row[4], row[5]] for row in auxiliary if row[0] == 'auxiliary shift']
    aux_block = table('auxiliary', 'Recovery with auxiliary-only candidate selection. Cells report means and sample standard deviations.',
                      ['Language and condition', 'Queries', 'Weighted (SD)', 'Macro (SD)'], aux_rows, [1.25, .65, 1.05, 1.05])
    shift_block = table('source-control', 'Recovery on fixed WMT23 clients with different auxiliary sources. Cells report means and sample standard deviations.',
                        ['Language and source', 'Queries', 'Weighted (SD)', 'Macro (SD)'], shift_rows, [1.25, .65, 1.05, 1.05])
    tex = tex.replace('Candidate selection based only on auxiliary documents was evaluated in each preprocessing condition.',
                      r'Table~\ref{tab:auxiliary} reports recovery with candidate selection based only on auxiliary documents.')
    tex = tex.replace('(Supplementary Table S18)', r'(Table~\ref{tab:source-control})')
    marker = 'Out-of-vocabulary queries remained outside the generated query universe.'
    tex = tex.replace(marker, marker + '\n\n' + aux_block + '\n' + shift_block)

    controls = [[row[0].title() + r'\newline ' + row[1] + ' ' + row[2], row[3], row[4], row[5]] for row in source_rows(supplement, 21)]
    control_block = table('candidate-control', 'Controlled-candidate and shuffled-rank recovery. Cells report means and sample standard deviations.',
                          ['Control and condition', 'Queries', 'Weighted (SD)', 'Macro (SD)'], controls, [1.25, .65, 1.05, 1.05])
    tex = tex.replace('Expanded controlled-candidate and shuffled-rank means are reported in Supplementary Table S21.', '')
    tex = tex.replace('(Supplementary Table S21)', r'(Table~\ref{tab:candidate-control})')
    tex = tex.replace('The expanded shuffled-rank control also used \\ExpandedSeeds{} seeds '+r'(Table~\ref{tab:candidate-control}).',
                      r'The expanded shuffled-rank control also used \ExpandedSeeds{} seeds.' + '\n\n' + control_block)

    rule_rows = [[row[0], row[1], row[2], row[4]] for row in source_rows(supplement, 7)]
    rules = table('rules', 'Mizo suffix-deletion rules. Exceptions refer to the current form. Minimum lengths apply to the remaining stem.',
                  ['Suffix', 'Priority', 'Minimum letters', 'Exceptions'], rule_rows, [.6, .6, .7, 2.1])
    tex = tex.replace('Supplementary Tables S7 and S8 supplied the complete rules and excluded stopword lists.',
                      r'Table~\ref{tab:rules} specifies the suffix rules. Each eligible suffix was deleted. The frozen stopword lists were retained in the reconstruction records.')
    marker = 'The annotation disagreements were retained in the supplementary material.'
    tex = tex.replace(marker, 'The reviewed labels and disagreement records were retained in the reconstruction records.\n\n' + rules)

    replacements = {
        '(Supplementary Table S10)': '',
        'Earlier three-seed summaries were retained in Supplementary Tables S11, S14 and S15. ': '',
        '(Supplementary Table S16)': r'(Figure~\ref{fig:partition})',
        '(Supplementary Table S17)': '',
        '(Supplementary Table S19)': '',
        '(Supplementary Table S20)': '',
        'Supplementary Table S16 reports the expanded recovery means and sample standard deviations. Earlier three-seed summaries were retained in Supplementary Tables S13 and S14. ': '',
        'The earlier \\SensitivitySeeds{}-seed estimates were retained in Supplementary Table S12. ': '',
        '(Supplementary Table S5)': '',
        'Earlier three-seed results were retained in Supplementary Tables S11-S15. ': '',
        'Supplementary Tables S9-S15 supplied diagnostics and descriptive sensitivity summaries.': 'Exact configurations, vocabulary diagnostics and exhaustive seed-level summaries were retained in the reconstruction records.',
        'The supplementary material contained the full comparison tables and source-data identifiers.': 'Full comparison tables and source-data identifiers were retained in the reconstruction records.',
        ' and Supplementary Table S16': '',
        ' and Supplementary Table S9': '',
        'Production-scale likelihood and assignment-objective audits are reported in Supplementary Table S19.': r'The production-scale audit covered \IndependentAuditRuns{} runs and \IndependentAuditUpdates{} updates with a maximum relative cost difference of \IndependentAuditMaxRelative{}.',
    }
    for old, new in replacements.items():
        tex = tex.replace(old, new)
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/expanded_recovery.json').read_text(encoding='utf-8'))
    tex = tex.replace(r'Partition-sensitivity seeds were \SensitivityPartitionSeeds{}. Attack seeds were \SensitivityAttackSeeds{}.',
                      'Expanded partition seeds were ' + ', '.join(map(str, config['partition_seeds'])) +
                      '. Each expanded condition used attack seeds ' + str(min(config['attack_seeds'])) + '-' + str(max(config['attack_seeds'])) + '.')
    tex = re.sub(r'\\bmhead\*\{Supplementary information\}.*?(?=\\section\*\{Declarations\})', '', tex, flags=re.S)
    tex = tex.replace('The complete recorded results and reconstruction scripts are provided in Online Resource 2 (reproducibility-release.zip).',
                      'The expanded recorded results and reconstruction scripts were assembled in reproducibility-release.zip. Public deposition of the expanded reconstruction records remained pending.')
    availability = (
        r'The recorded results, reviewed preprocessing resources, configurations and reconstruction scripts were deposited at \url{' + RELEASE_URL +
        r'}~\cite{studydata}. The release asset reproducibility-release.zip contains the expanded records and source identifiers. '
        'Numerical reconstruction from recorded results requires no raw corpus text. '
        r'Full attack reruns require the original WMT23 IndicNE-Corp, FLORES-200 and Enron datasets~\cite{indic,flores,enrondata}, '
        'together with the IHOP code revision specified in Methods. Dataset acquisition instructions and file identifiers are included in the archive. '
        'Raw corpus text and private review correspondence were excluded from redistribution.\n\n')
    tex = re.sub(r'(?<=\\subsection\*\{Data availability\}\n).*?(?=\\bibliography\{)', lambda match: availability, tex, flags=re.S)
    tex = re.sub(r' +([.,])', r'\1', tex)
    if 'Supplementary Table' in tex or 'Online Resource' in tex:
        raise ValueError('A supplementary dependency remained in the article')
    return preamble + r'\begin{document}' + tex
