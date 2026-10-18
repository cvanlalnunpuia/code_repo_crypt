from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np

BASE = 'results/corpus/expanded-recovery/'
METRICS = ['query_weighted_recovery', 'distinct_keyword_macro_recovery']


def load_expanded(root, sources):
    status = root/BASE/'status.json'
    if not status.exists() or json.loads(status.read_text(encoding='utf-8')).get('stage') != 'complete':
        return None
    result = {}
    for key, filename in [('analysis','expanded-analysis.json'), ('statistics','extended-statistics.json'),
                          ('validation','independent-comparison.json'), ('controls','auxiliary-controls.json')]:
        path = root/BASE/filename
        sources[BASE+filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        result[key] = json.loads(path.read_text(encoding='utf-8'))
    for name in ['status.json', 'manifest.json', 'checkpoint.json']:
        path = root/BASE/name
        sources[BASE+name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['summary-format-provenance.json', 'figure-data.json']:
        path = root/BASE/name
        if path.exists():
            sources[BASE+name] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources['configs/expanded_recovery.json'] = hashlib.sha256((root/'configs/expanded_recovery.json').read_bytes()).hexdigest()
    for name in ['scripts/expanded_manuscript.py', 'scripts/expanded_statistics.py',
                 'scripts/independent_ihop.py', 'scripts/audit_production_ihop.py', 'scripts/auxiliary_controls.py',
                 'scripts/expanded_partition_figure.py']:
        sources[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    result['original'] = json.loads((root/'results/corpus/production-recovery-analysis.json').read_text(encoding='utf-8'))
    checkpoint = json.loads((root/BASE/'checkpoint.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/BASE/'manifest.json').read_text(encoding='utf-8'))
    if result['analysis']['checkpoint_sha256'] != sources[BASE+'checkpoint.json']:
        raise ValueError('Expanded analysis checkpoint changed')
    if checkpoint['manifest_sha256'] != sources[BASE+'manifest.json'] or len(checkpoint['runs']) != len(manifest['tasks']):
        raise ValueError('Expanded execution is incomplete or differs from its manifest')
    groups = defaultdict(list)
    for run in checkpoint['runs'].values():
        groups[tuple(run['task'][key] for key in ['family','partition','language','condition','distribution'])].append(run)
    for row in result['analysis']['summaries']:
        values = sorted(groups[tuple(row[key] for key in ['family','partition','language','condition','distribution'])], key=lambda run:run['seed'])
        if row['seeds'] != [value['seed'] for value in values]:
            raise ValueError('Expanded summary seed coverage differs')
        for metric in METRICS:
            observed = [value['scores'][-1][metric] for value in values]
            if row[metric]['mean'] != float(np.mean(observed)):
                raise ValueError('Expanded recovery mean differs from its saved runs')
            if len(observed)>1 and row[metric]['sd'] != float(np.std(observed, ddof=1)):
                raise ValueError('Expanded recovery standard deviation differs')
            if len(observed)==1:
                row[metric]['sd'] = None
    from expanded_statistics import original_extension_statistics
    if result['statistics'] != original_extension_statistics(list(checkpoint['runs'].values()), manifest['config']):
        raise ValueError('Extended comparison statistics differ from complete reconstruction')
    baseline = json.loads((root/'results/corpus/production-fixed-split-experiments.json').read_text(encoding='utf-8'))
    baseline_lookup = {(value['dataset_id'], value['distribution'], value['seed']):value for value in baseline['runs']}
    for comparison in result['validation']['comparisons']:
        key = (comparison['dataset_id'], comparison['distribution'], comparison['seed'])
        run = checkpoint['runs']['independent-'+key[0]+':'+key[1]+':'+str(key[2])]
        reference = baseline_lookup[key]
        error = max(abs(a[metric]-b[metric]) for a,b in zip(run['scores'],reference['scores']) for metric in METRICS)
        if error != comparison['maximum_score_difference'] or comparison['cost_audit'] != run['cost_audit']:
            raise ValueError('Production audit summary differs from its saved run')
    result['lookup'] = {tuple(row[key] for key in ['family','partition','language','condition','distribution']):row
                        for row in result['analysis']['summaries']}
    figure_path = root/BASE/'figure-data.json'
    if figure_path.exists():
        from expanded_partition_figure import data as figure_data
        if json.loads(figure_path.read_text(encoding='utf-8')) != figure_data():
            raise ValueError('Expanded figure data differed from its saved inputs')
    if result['analysis']['status'] != 'passed' or not result['validation']['comparisons']:
        raise ValueError('Expanded recovery validation is incomplete')
    return result


def expanded_facts(expanded, fact):
    analysis, validation = expanded['analysis'], expanded['validation']
    keys = [(row['condition'], row['distribution'], row['metric']) for row in expanded['original']['language_comparisons']
            if row['holm_adjusted_p'] < .05]
    positive = 0
    for partition in analysis['config']['partition_seeds']:
        for condition, distribution, metric in keys:
            left = expanded['lookup'][('partition', partition, 'lus', condition, distribution)][metric]['mean']
            right = expanded['lookup'][('partition', partition, 'en', condition, distribution)][metric]['mean']
            positive += left > right
    values = {
        'ExpandedPartitions': len(analysis['config']['partition_seeds']),
        'ExpandedSeeds': len(analysis['config']['attack_seeds']),
        'ExtendedSeeds': len(expanded['statistics']['seeds']),
        'ExpandedPositiveDirections': positive,
        'ExpandedDirectionCount': len(keys)*len(analysis['config']['partition_seeds']),
        'ExpandedRunCount': analysis['run_count'],
        'ExtendedLanguageSignificant': sum(row['holm_adjusted_p']<.05 for row in expanded['statistics']['language_comparisons']),
        'IndependentAuditRuns': len(validation['comparisons']),
        'IndependentAuditMatched': sum(row['matched'] for row in validation['comparisons']),
        'IndependentAuditUpdates': sum(row['cost_audit']['updates_checked'] for row in validation['comparisons']),
    }
    for name, value in values.items():
        fact(name, int(value), BASE+'expanded-analysis.json', 'expanded_manuscript.expanded_facts.'+name)
    fact('ExtendedMinimumP', expanded['statistics']['minimum_two_sided_probability'], BASE+'extended-statistics.json',
         'minimum_two_sided_probability', 8)
    maximum = max(row['cost_audit']['maximum_relative_cost_error'] for row in validation['comparisons'])
    fact('IndependentAuditMaxRelative', f'{maximum:.3e}', BASE+'independent-comparison.json',
         'maximum comparisons.cost_audit.maximum_relative_cost_error')
    lookup = expanded['lookup']
    partition = next(row['partition'] for row in analysis['summaries'] if row['family']=='controlled')
    for language, prefix in [('en', 'EN'), ('lus', 'LUS')]:
        stem = lookup[('controlled', partition, language, 'stemming', 'zipf_iid')][METRICS[0]]['mean']
        stop = lookup[('controlled', partition, language, 'stopword_removal', 'zipf_iid')][METRICS[0]]['mean']
        fact('Expanded'+prefix+'ControlledDifference', stem-stop, BASE+'expanded-analysis.json',
             'controlled '+language+' Zipf weighted stemming minus stopwords', 4)
        for source, suffix in [('wmt23', 'WMT'), ('flores200', 'Flores')]:
            mean = lookup[('auxiliary_shift', partition, language, source, 'zipf_iid')][METRICS[0]]['mean']
            fact('Expanded'+prefix+'Aux'+suffix, mean, BASE+'expanded-analysis.json',
                 'auxiliary_shift '+language+' '+source+' Zipf weighted', 4)
    differences = [lookup[('balanced', seed, 'en', 'flores200', 'zipf_iid')][METRICS[0]]['mean']-
                   lookup[('balanced', seed, 'en', 'wmt23', 'zipf_iid')][METRICS[0]]['mean']
                   for seed in analysis['config']['partition_seeds']]
    for suffix, value in [('Min', min(differences)), ('Max', max(differences))]:
        fact('ExpandedBalanced'+suffix, value, BASE+'expanded-analysis.json', 'expanded English Zipf weighted FLORES minus WMT23 '+suffix, 4)


def incorporate_main(tex):
    start = tex.index(r'\begin{abstract}')+len(r'\begin{abstract}')
    end = tex.index(r'\end{abstract}')
    tex = tex[:start]+r'''
Partition sensitivity in statistical query recovery was evaluated using English-Mizo text. Improved Statistical Query Recovery against Searchable Symmetric Encryption through Quadratic Optimization (IHOP) was applied to \ProductionPairs{} constructed text-block pairs. Expanded analyses used \ExpandedPartitions{} partitions and \ExpandedSeeds{} attack seeds. Original positive language contrasts retained positive means in \ExpandedPositiveDirections{} of \ExpandedDirectionCount{} expanded comparisons. Controlled-candidate stemming contrasts were close to zero. Auxiliary-source changes produced different recovery responses across languages. Production-scale likelihood and assignment-objective checks covered \IndependentAuditRuns{} runs. Recovery comparisons remained conditional on document boundaries, partition membership and candidate construction.
'''+tex[end:]
    marker = '@@PARTITION_FIGURE@@'
    paragraph = r'''

Expanded recovery analyses used \ExpandedPartitions{} paired partitions and \ExpandedSeeds{} attack seeds per condition (Supplementary Table S16). The original positive language contrasts retained positive means in \ExpandedPositiveDirections{} of \ExpandedDirectionCount{} expanded comparisons. The original-partition extension used \ExtendedSeeds{} attack seeds. Its exploratory language family contained \ExtendedLanguageSignificant{} Holm-adjusted probabilities below \Alpha{} (Supplementary Table S17). Its minimum possible two-sided probability was \ExtendedMinimumP{}. The original confirmatory families were preserved separately.

Candidate selection based only on auxiliary documents was evaluated in each preprocessing condition. A separate control changed the auxiliary corpus while retaining the WMT23 client documents and a shared candidate set (Supplementary Table S18). English Zipf query-weighted recovery was \ExpandedENAuxWMT{} with WMT23 auxiliary data and \ExpandedENAuxFlores{} with FLORES-200 auxiliary data. The corresponding Mizo means were \ExpandedLUSAuxWMT{} and \ExpandedLUSAuxFlores{}. Equal auxiliary counts were used in that comparison. Both controls excluded client data from candidate ranking. Candidate labels absent from client documents were retained. Out-of-vocabulary queries remained outside the generated query universe.

Independent likelihood and assignment-objective calculations were checked at \IndependentAuditUpdates{} updates across \IndependentAuditRuns{} original production runs (Supplementary Table S19). Audited reruns agreed with the recorded recovery scores in \IndependentAuditMatched{} runs. The maximum relative cost difference was \IndependentAuditMaxRelative{}. Assignment objectives were checked within rounding bounds derived from the measured cost differences. The SciPy assignment routine was shared. Separate volume-only trajectories used independently computed costs (Supplementary Table S20). Equal-objective assignments permitted different recovered labels.
'''
    tex = tex.replace(marker, marker+paragraph)
    tex = tex.replace('Production-scale independent equivalence remained unvalidated.',
                      'Production-scale likelihood and assignment-objective checks are reported in Supplementary Table S19.')
    tex = tex.replace('Unknown candidate vocabularies and auxiliary-distribution mismatch remained untested.',
                      'Auxiliary-source controls are reported in Supplementary Table S18. Out-of-vocabulary queries and empirical query logs remained untested.')
    tex = tex.replace('Source composition and the selected vocabulary still differed between languages.',
                      'The paired texts shared source membership. Their selected vocabularies differed. Translation differences remained part of the language comparison.')
    tex = tex.replace('Additional conditions used \\SensitivitySeeds{} attack seeds, which limited assessment of within-partition variation.',
                      'The expanded conditions used \\ExpandedSeeds{} attack seeds. The original partition extension used \\ExtendedSeeds{} seeds.')
    tex = tex.replace('Independent production-scale attack equivalence remained unvalidated.',
                      'Production-scale likelihood and assignment-objective audits are reported in Supplementary Table S19. The SciPy assignment routine was shared.')
    tex = tex.replace('Production-scale likelihood and assignment-objective checks are reported in Supplementary Table S19. The production experiment',
                      'Production-scale likelihood and assignment-objective checks were also completed. The production experiment')
    methods = r'''

The expanded design used \ExpandedPartitions{} partition seeds recorded in the frozen configuration. Each condition used \ExpandedSeeds{} consecutive attack seeds beginning at zero. The original partition was extended to \ExtendedSeeds{} seeds beginning at zero. Partition, length-balanced, controlled-candidate and shuffled-rank settings were retained. The frozen task manifest contained \ExpandedRunCount{} runs. Previously completed runs were reused only after their dataset hashes and endpoints matched. The original confirmatory families were unchanged. The original-partition extension used separate exploratory language and preprocessing families with the same paired procedures.

Auxiliary-only candidates were ranked by auxiliary document frequency. Eligibility required non-zero, non-complete auxiliary support. Client support was unrestricted. Auxiliary-source controls used the same WMT23 client partition with WMT23 or FLORES-200 auxiliary excerpts. Auxiliary and client counts followed the matched-size configuration in Table~\ref{tab:corpora}. A common candidate set was selected using both auxiliary sets without client ranking. These controls used tokenisation only. Candidate labels absent from the client documents were retained.

Production audits independently constructed binary incidence probabilities and binomial likelihood costs at every original update. Maximum entrywise cost differences and assignment-objective differences were recorded. The objective tolerance was twice the assignment row count times the maximum cost difference, plus $10^{-7}$. Both implementations used the SciPy linear assignment routine. Separate volume-only trajectories used independently computed costs at attack seed 0. Their recovery values described single trajectories. Between-seed standard deviations were unmeasured for those trajectories.
'''
    tex = tex.replace('An independently selected trajectory was excluded from this audit.',
                      'An independently selected trajectory was excluded from this audit.'+methods)
    tex = tex.replace('Their direction was retained in \\SensitivityPositiveDirections{} of \\SensitivityDirectionCount{} additional comparisons (Figure~\\ref{fig:partition}).',
                      'Their direction was retained in \\ExpandedPositiveDirections{} of \\ExpandedDirectionCount{} expanded comparisons (Figure~\\ref{fig:partition} and Supplementary Table S16).')
    tex = tex.replace('The recorded experimental results and baseline reconstruction scripts were deposited at',
                      'The original experimental results and baseline reconstruction scripts were deposited at')
    tex = tex.replace('The archive contains reviewed linguistic resources, configurations and numerical source identifiers.',
                      'The baseline archive contains reviewed linguistic resources, configurations and numerical source identifiers. The complete recorded results and reconstruction scripts are provided in Online Resource 2 (reproducibility-release.zip).')
    tex = tex.replace('Equal-objective assignments permitted different recovered labels.',
                      'Equal-objective assignments permitted different recovered labels. Expanded controlled-candidate and shuffled-rank means are reported in Supplementary Table S21.')
    import re
    tex = re.sub(r'Partition sensitivity analyses used .*?Statistical significance was not tested in those comparisons\.',
        r'Figure~\\ref{fig:partition} shows selected paired contrasts across the expanded partitions. The original partition and additional partitions used the same \\ExpandedSeeds{} attack seeds. Earlier three-seed summaries were retained in Supplementary Tables S11, S14 and S15. Statistical significance was not tested in those comparisons.', tex, flags=re.S)
    tex = tex.replace('English length-balanced Zipf query-weighted FLORES-minus-WMT23 differences ranged from \\BalancedSensitivityMin{} to \\BalancedSensitivityMax{} at \\Keywords{} keywords across additional partitions (Figure~\\ref{fig:partition}). Supplementary Tables S13 and S14 report recovery means and paired contrasts with sample standard deviations.',
        'English length-balanced Zipf query-weighted FLORES-minus-WMT23 differences ranged from \\ExpandedBalancedMin{} to \\ExpandedBalancedMax{} at \\Keywords{} keywords across the expanded partitions (Figure~\\ref{fig:partition}). Supplementary Table S16 reports the expanded recovery means and sample standard deviations. Earlier three-seed summaries were retained in Supplementary Tables S13 and S14.')
    tex = re.sub(r'A controlled candidate set retained .*?These \\SensitivityRuns\{\} runs were descriptive sensitivity analyses\.',
        lambda match: r'A controlled candidate set retained \ControlledKeywords{} word labels per language with distinct mapped stems. Tokenisation-only and stopword-only incidence matrices were identical on these retained terms. The stemming comparison incorporated incidences of all forms mapped to each selected stem. Across \ExpandedSeeds{} seeds, Zipf query-weighted stemming-minus-stopword differences were \ExpandedENControlledDifference{} in English and \ExpandedLUSControlledDifference{} in Mizo (Supplementary Table S21). Both differences were close to zero. The earlier \SensitivitySeeds{}-seed estimates were retained in Supplementary Table S12. The expanded shuffled-rank control also used \ExpandedSeeds{} seeds (Supplementary Table S21).', tex, flags=re.S)
    tex = tex.replace('The controlled word-to-stem incidence comparison had a positive English Zipf stemming-minus-stopword difference and a negative Mizo difference (Supplementary Table S12).',
        'The controlled word-to-stem incidence comparison produced Zipf stemming-minus-stopword differences close to zero across \\ExpandedSeeds{} attack seeds (Supplementary Table S21). The signs matched the earlier three-seed estimates. Their magnitudes were smaller in the expanded evaluation.')
    tex = tex.replace('The English length-balanced Zipf recovery difference changed sign across additional partitions (Supplementary Table S13).',
        'The English length-balanced Zipf recovery difference changed sign across the expanded partitions (Figure~\\ref{fig:partition} and Supplementary Table S16).')
    tex = tex.replace('Auxiliary-source controls are reported in Supplementary Table S18. Out-of-vocabulary queries',
        'Changing the auxiliary source reduced Mizo Zipf recovery and increased English Zipf recovery in the matched control (Supplementary Table S18). The direction of the auxiliary-source effect depended on the evaluated language and matrices. Out-of-vocabulary queries')
    tex = tex.replace('The original partition means in Figure~\\ref{fig:partition} used those same attack seeds.',
        'Earlier three-seed results were retained in Supplementary Tables S11-S15. Figure~\\ref{fig:partition} used \\ExpandedSeeds{} matched attack seeds for the original partition and each expanded partition.')
    return tex


def supplementary_tables(expanded, long_table, parts, language_names, condition_names, distribution_names):
    analysis = expanded['analysis']
    def measurement(row, metric):
        return f"{row[metric]['mean']:.4f} ({row[metric]['sd']:.4f})"
    parts.append(r'Tables~\ref{tab:supp:16}, \ref{tab:supp:17}, \ref{tab:supp:18}, \ref{tab:supp:19}, \ref{tab:supp:20} and \ref{tab:supp:21} report the expanded recovery evaluation.')
    rows = []
    for row in analysis['summaries']:
        if row['family'] in ['partition', 'balanced']:
            rows.append([row['partition'], language_names[row['language']],
                condition_names.get(row['condition'], {'flores200':'FLORES','wmt23':'WMT23'}.get(row['condition'], row['condition'])), distribution_names[row['distribution']],
                measurement(row, METRICS[0]), measurement(row, METRICS[1])])
    long_table('Expanded partition and corpus recovery means with sample standard deviations.',
               ['Partition', 'Language', 'Condition or corpus', 'Queries', 'Weighted (SD)', 'Macro (SD)'], rows, 'llllrr')
    rows = []
    for family in ['language_comparisons', 'preprocessing_comparisons']:
        for row in expanded['statistics'][family]:
            label = ('Mizo minus English '+condition_names[row['condition']] if family=='language_comparisons' else
                     language_names[row['language']]+' '+condition_names[row['left']]+' minus '+condition_names[row['right']])
            low, high = row['confidence_interval']
            rows.append([label, distribution_names[row['distribution']], 'Weighted' if row['metric']==METRICS[0] else 'Macro',
                         f"{row['mean']:.4f}", f'[{low:.4f}, {high:.4f}]', f"{row['holm_adjusted_p']:.8f}"])
    long_table('Exploratory original-partition contrasts with extended attack seeds.',
               ['Contrast', 'Queries', 'Metric', 'Difference', '95\\% CI', 'Holm $p$'], rows, 'llllrr')
    parts.append('Confidence intervals were unadjusted. Exploratory language and preprocessing comparisons were adjusted within separate families. The original confirmatory comparisons were preserved.')
    rows = []
    for row in analysis['summaries']:
        if row['family'] in ['auxiliary_only', 'auxiliary_shift']:
            rows.append([row['family'].replace('_',' '), language_names[row['language']],
                         condition_names.get(row['condition'], row['condition']), distribution_names[row['distribution']],
                         measurement(row, METRICS[0]), measurement(row, METRICS[1])])
    long_table('Recovery with auxiliary-only candidate selection and auxiliary-source controls.',
               ['Control', 'Language', 'Condition or source', 'Queries', 'Weighted (SD)', 'Macro (SD)'], rows, 'llllrr')
    parts.append(expanded['controls']['scope'])
    grouped = {}
    for row in expanded['validation']['comparisons']:
        grouped.setdefault((row['dataset_id'], row['distribution']), []).append(row)
    rows = []
    for (dataset, distribution), values in sorted(grouped.items()):
        language, condition = dataset.split(':')
        rows.append([language_names[language], condition_names[condition], distribution_names[distribution], len(values),
                     sum(row['cost_audit']['updates_checked'] for row in values),
                     f"{max(row['cost_audit']['maximum_relative_cost_error'] for row in values):.3e}",
                     f"{max(row['maximum_score_difference'] for row in values):.3e}"])
    long_table('Production-scale likelihood, assignment-objective and checkpoint-score checks.',
               ['Language', 'Condition', 'Queries', 'Runs', 'Updates', 'Maximum relative cost error', 'Maximum score difference'], rows, 'llllrrr')
    parts.append(expanded['validation']['scope'])
    rows = []
    for row in analysis['summaries']:
        if row['family']=='independent_trajectory':
            rows.append([language_names[row['language']], condition_names[row['condition']], distribution_names[row['distribution']],
                         f"{row[METRICS[0]]['mean']:.4f}", f"{row[METRICS[1]]['mean']:.4f}"])
    long_table('Separate volume-only trajectory recovery on the first original attack seed.',
               ['Language', 'Condition', 'Queries', 'Weighted', 'Macro'], rows, 'lllrr')
    parts.append('The separate implementation used a shared SciPy assignment routine. Likelihood arithmetic differed. Different equal-objective assignments were permitted. Recovery scores described the resulting trajectories.')
    rows = []
    for row in analysis['summaries']:
        if row['family'] in ['controlled', 'shuffled']:
            rows.append([row['family'], language_names[row['language']], condition_names[row['condition']],
                         distribution_names[row['distribution']], measurement(row, METRICS[0]), measurement(row, METRICS[1])])
    if not rows:
        raise ValueError('Expanded candidate and rank controls were absent')
    long_table('Expanded controlled-candidate and shuffled-rank recovery means with sample standard deviations.',
               ['Control', 'Language', 'Condition', 'Queries', 'Weighted (SD)', 'Macro (SD)'], rows, 'llllrr')
