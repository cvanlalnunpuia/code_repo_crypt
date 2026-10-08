from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import scipy
from springer_template import OFFICIAL, migrate, bibliography

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'paper'
CONDITIONS = ['tokenisation_only', 'stopword_removal', 'stemming']
LABELS = {'tokenisation_only': 'Tokenisation', 'stopword_removal': 'Stopwords', 'stemming': 'Stemming'}
LANG = {'en': 'English', 'lus': 'Mizo'}
DIST = {'uniform_iid': 'Uniform', 'zipf_iid': 'Zipf'}
METRIC = {'query_weighted_recovery': 'Weighted', 'distinct_keyword_macro_recovery': 'Macro'}

REFERENCES = [
    ('curtmola', 'R. Curtmola, J. Garay, S. Kamara, and R. Ostrovsky', 'Searchable Symmetric Encryption: Improved Definitions and Efficient Constructions', 'Cryptology ePrint Archive, 2006/210', 'https://eprint.iacr.org/2006/210'),
    ('islam', 'M. S. Islam, M. Kuzu, and M. Kantarcioglu', 'Access Pattern Disclosure on Searchable Encryption: Ramification, Attack and Mitigation', 'NDSS, 2012', 'https://www.ndss-symposium.org/ndss2012/ndss-2012-programme/access-pattern-disclosure-searchable-encryption-ramification-attack-and-mitigation/'),
    ('blackstone', 'L. Blackstone, S. Kamara, and T. Moataz', 'Revisiting Leakage Abuse Attacks', 'NDSS, 2020', 'https://www.ndss-symposium.org/ndss-paper/revisiting-leakage-abuse-attacks/'),
    ('leaker', 'S. Kamara, A. Kati, T. Moataz, T. Schneider, A. Treiber, and M. Yonli', 'SoK: Cryptanalysis of Encrypted Search with LEAKER - A Framework for Leakage Attack Evaluation on Real-World Data', 'IEEE European Symposium on Security and Privacy, 2022', 'https://eprint.iacr.org/2021/1035'),
    ('ihop', 'S. Oya and F. Kerschbaum', 'IHOP: Improved Statistical Query Recovery against Searchable Symmetric Encryption through Quadratic Optimization', 'USENIX Security, 2022', 'https://www.usenix.org/conference/usenixsecurity22/presentation/oya'),
    ('indic', 'S. Pal, P. Pakray, S. R. Laskar, L. Laitonjam, V. Khenglawt, S. Warjri, P. K. Dadure, and S. K. Dash', 'Findings of the WMT 2023 Shared Task on Low-Resource Indic Language Translation', 'Eighth Conference on Machine Translation, 2023', 'https://aclanthology.org/2023.wmt-1.56/'),
    ('flores', 'NLLB Team', 'No Language Left Behind: Scaling Human-Centered Machine Translation', 'arXiv:2207.04672, 2022', 'https://arxiv.org/abs/2207.04672'),
    ('floresrepo', 'Meta', 'FLORES-200 dataset documentation and licence', 'Dataset documentation', 'https://github.com/facebookresearch/flores/blob/main/flores200/README.md'),
    ('ihopcode', 'S. Oya', 'IHOP evaluation implementation', 'Software repository', 'https://github.com/simon-oya/ihop-code'),
    ('snowball', 'Snowball project', 'The English (Porter2) stemming algorithm', 'Algorithm documentation', 'https://snowballstem.org/algorithms/english/stemmer.html'),
    ('enrondata', 'Carnegie Mellon University', 'Enron Email Dataset', 'Dataset documentation', 'https://www.cs.cmu.edu/~enron/'),
    ('cash', 'D. Cash, P. Grubbs, J. Perry, and T. Ristenpart', 'Leakage-Abuse Attacks Against Searchable Encryption', 'ACM CCS, 2015, pp. 668-679', 'https://doi.org/10.1145/2810103.2813700'),
    ('damie', 'M. Damie, F. Hahn, and A. Peter', 'A Highly Accurate Query-Recovery Attack against Searchable Encryption using Non-Indexed Documents', 'USENIX Security, 2021, pp. 143-160', 'https://www.usenix.org/conference/usenixsecurity21/presentation/damie'),
    ('sap', 'S. Oya and F. Kerschbaum', 'Hiding the Access Pattern is Not Enough: Exploiting Search Pattern Leakage in Searchable Encryption', 'USENIX Security, 2021, pp. 127-142', 'https://www.usenix.org/conference/usenixsecurity21/presentation/oya'),
    ('hoover', "A. Hoover, R. Ng, D. Khu, Yao'An Li, J. Lim, D. Ng, Jed Lim, and Y. Song", 'Leakage-Abuse Attacks Against Structured Encryption for SQL', 'USENIX Security, 2024, pp. 7411-7428', 'https://www.usenix.org/conference/usenixsecurity24/presentation/hoover'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(relative, sources):
    path = ROOT / relative
    sources[relative] = digest(path)
    return json.loads(path.read_text(encoding='utf-8'))


def escape(text):
    return str(text).replace('&', r'\&').replace('_', r'\_').replace('%', r'\%').replace('ṭ', r'\d{t}')


def table(caption, label, headers, rows, columns):
    body = '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)
    return (r'\begin{table}[htbp]' + '\n' + r'\centering\small' + '\n' +
            f'\\caption{{{caption}}}\\label{{{label}}}\n' +
            f'\\begin{{tabular}}{{@{{}}{columns}@{{}}}}\n\\toprule\n' +
            ' & '.join(headers) + r' \\' + '\n\\midrule\n' + body + '\n\\bottomrule\n\\end{tabular}\n\\end{table}')


def interval(values):
    return f'[{values[0]:.4f}, {values[1]:.4f}]'


def recovery_figure(result):
    colours = {'tokenisation_only': 'blue', 'stopword_removal': 'orange', 'stemming': 'teal'}
    symbols = {'tokenisation_only':r'\bullet','stopword_removal':r'\square','stemming':r'\triangle'}
    lines = [r'\begin{figure}[htbp]', r'\centering']
    for language in ['en', 'lus']:
        for distribution in ['uniform_iid', 'zipf_iid']:
            lines.extend([r'\begin{minipage}{0.48\textwidth}\centering', r'\setlength{\unitlength}{1pt}',
                          r'\begin{picture}(220,155)',
                          f'\\put(30,146){{\\makebox(0,0)[l]{{\\scriptsize ({"abcd"[["en","lus"].index(language)*2+["uniform_iid","zipf_iid"].index(distribution)]})}}}}',
                          r'\put(30,134){\makebox(0,0)[l]{\scriptsize Query-weighted recovery}}',
                          r'\put(30,28){\line(1,0){180}}', r'\put(30,28){\line(0,1){100}}'])
            for tick in [0, 0.5, 1]:
                y = 28 + tick * 100
                lines.append(f'\\put(24,{y:.2f}){{\\makebox(0,0)[r]{{\\scriptsize {tick:g}}}}}')
            checkpoints = sorted({item['checkpoint'] for item in result['aggregate']})
            xs = [30 + 180 * np.log10(c + 1) / np.log10(max(checkpoints) + 1) for c in checkpoints]
            for checkpoint, x in zip(checkpoints, xs):
                lines.append(f'\\put({x:.3f},18){{\\makebox(0,0){{\\scriptsize {checkpoint}}}}}')
            for condition in CONDITIONS:
                points = sorted([item for item in result['aggregate'] if item['language'] == language and item['distribution'] == distribution and item['condition'] == condition and item['metric'] == 'query_weighted_recovery'], key=lambda item: item['checkpoint'])
                ys = [28 + 100 * item['mean'] for item in points]
                lines.append(f'{{\\color{{{colours[condition]}}}')
                for x, y in zip(xs, ys):
                    lines.append(f'\\put({x:.3f},{y:.3f}){{\\makebox(0,0){{\\scriptsize ${symbols[condition]}$}}}}')
                for x1, x2, y1, y2 in zip(xs, xs[1:], ys, ys[1:]):
                    lines.append(f'\\qbezier({x1:.3f},{y1:.3f})({(x1+x2)/2:.3f},{(y1+y2)/2:.3f})({x2:.3f},{y2:.3f})')
                lines.append('}')
            lines.extend([r'\put(120,3){\makebox(0,0){\scriptsize IHOP iterations}}', r'\end{picture}', r'\end{minipage}'])
        lines.append(r'\par\vspace{4pt}')
    lines.extend([r'\small$\bullet$ Tokenisation\quad$\square$ Stopwords\quad$\triangle$ Stemming',
                  r'\caption{Mean query-weighted recovery by iteration. (a) English uniform. (b) English Zipf. (c) Mizo uniform. (d) Mizo Zipf. Symbols identify preprocessing}',
                  r'\label{fig:recovery}', r'\end{figure}'])
    return '\n'.join(lines)


def build():
    sources = {}
    production = load('results/corpus/production-recovery-analysis.json', sources)
    structure = load('results/corpus/export-structure-analysis.json', sources)
    balanced = load('results/corpus/length-balanced-comparison.json', sources)
    matched = load('results/corpus/flores200-wmt23-matched-comparison.json', sources)
    sensitivity = load('results/corpus/keyword-size-sensitivity-comparison.json', sources)
    revision_diagnostics = load('results/corpus/reviewer-sensitivity/diagnostics.json', sources)
    revision_controls = load('results/corpus/reviewer-sensitivity/controlled-keywords.json', sources)
    revision_results = load('results/corpus/reviewer-sensitivity/analysis.json', sources)
    from review_uncertainty import build as rebuild_uncertainty, figure_tex
    uncertainty = load('results/corpus/reviewer-sensitivity/uncertainty.json', sources)
    real_check = load('results/corpus/reviewer-sensitivity/real-assignment-check.json', sources)
    iterative = load('results/corpus/reviewer-sensitivity/iterative-check.json', sources)
    if not all(c['exhaustive_updates'] and c['exact_checkpoint_reconstruction'] and c['maximum_cost_error'] < 1e-8 for c in iterative['cases']):
        raise ValueError('Iterative update audit failed')
    load('results/corpus/flores200-length-balanced-experiments.json', sources)
    load('results/corpus/wmt23-length-balanced-experiments.json', sources)
    if not all(c['exhaustive_assignment_check'] and c['maximum_cost_error'] < 1e-8 for c in real_check['cases']):
        raise ValueError('Real-data assignment validation failed')
    if real_check['case_count'] != len(real_check['cases']):
        raise ValueError('Real-data validation case count differs')
    if uncertainty != rebuild_uncertainty():
        raise ValueError('Uncertainty summaries differ from checkpoint reconstruction')
    load('results/corpus/reviewer-sensitivity/checkpoint.json', sources)
    revision_config = load('configs/reviewer_sensitivity.json', sources)
    production_split_config = load('configs/corpus_split_production.json', sources)
    matched_selection_config = load('configs/wmt23_flores200_matched_control.json', sources)
    balanced_split_config = load('configs/length_balanced_control.json', sources)
    if revision_results['status'] != 'passed':
        raise ValueError('Sensitivity analyses are incomplete')
    ablation = load('results/unified/enron-ablation-summary.json', sources)
    enron = load('results/ihop_enron_baseline/aggregate.json', sources)
    smoke = load('results/ihop-enron-volume-seed0-py39-hash0.json', sources)
    stemming = load('results/corpus/stemmer-execution-evaluation.json', sources)
    clarification = load('results/corpus/linguistic-decisions.json', sources)
    if set(clarification) != {'source_sha256', 'mizo_error_types', 'scope'}:
        raise ValueError('Structured linguistic decisions contain unexpected fields')
    execution_policy = load('configs/mizo_stemmer_execution.json', sources)
    reuse = load('results/corpus/clarified-recovery-reuse.json', sources)
    if execution_policy['clarification_source_sha256'] != clarification['source_sha256']:
        raise ValueError('Clarification evidence differs from execution policy')
    production_manifest = load('results/corpus/ihop/manifest.json', sources)
    if reuse['identity']['manifest_sha256'] != sources['results/corpus/ihop/manifest.json']:
        raise ValueError('Clarification reuse manifest differs')
    review = load('results/corpus/returned-linguistic-review-audit.json', sources)
    audit = load('results/corpus/wmt23-audit.json', sources)
    wmt_config = load('configs/wmt23_indicnecorp.json', sources)
    experiment = load('results/corpus/production-fixed-split-experiments.json', sources)
    replay = load('results/corpus/production-independent-replay.json', sources)
    if experiment != replay:
        raise ValueError('Production replay differs')
    if production['inputs']['experiment_sha256'] != sources['results/corpus/production-fixed-split-experiments.json']:
        raise ValueError('Analysis experiment digest changed')
    significant = {(x['condition'], x['distribution'], x['metric']) for x in production['language_comparisons'] if x['holm_adjusted_p'] < 0.05}
    expected = {('stopword_removal', 'zipf_iid', 'query_weighted_recovery')} | {
        ('stemming', distribution, metric) for distribution in ['uniform_iid', 'zipf_iid']
        for metric in ['query_weighted_recovery', 'distinct_keyword_macro_recovery']}
    if significant != expected:
        raise ValueError('The named language findings require revised manuscript prose')
    if any(x['mean'] <= 0 for x in production['language_comparisons'] if x['holm_adjusted_p'] < 0.05):
        raise ValueError('The reported language direction requires revised prose')
    if any(x['holm_adjusted_p'] < 0.05 for x in production['preprocessing_comparisons'] if x['left'] == 'stemming' and x['right'] == 'stopword_removal'):
        raise ValueError('The incremental-stemming finding requires revised prose')
    if any(x['language'] != 'en' for x in balanced['comparisons'] if x['holm_adjusted_p'] < 0.05):
        raise ValueError('The balanced-corpus language finding requires revised prose')
    endpoints = {x['id']: x['checkpoints']['1000']['paired_difference']['mean'] for x in ablation['contrasts']}
    if min(endpoints, key=endpoints.get) != 'sparsity':
        raise ValueError('The named largest endpoint reduction requires revised prose')
    tokenisation = load('configs/tokenisation_only.json', sources)
    rules = load('resources/stemmers/lus-production.reviewed.json', sources)
    interventions = {name: load(f'configs/{name}_sweep.json', sources) for name in
                     ['keyword_size', 'access_collision', 'access_similarity', 'frequency_ambiguity', 'cooccurrence_similarity', 'sparsity']}
    facts = {}
    def fact(name, value, source, selector, precision=None):
        rendered = str(value) if precision is None else f'{value:.{precision}f}'
        facts[name] = {'value': value, 'rendered': rendered, 'source': source, 'selector': selector}
    lock = ROOT / 'requirements-2022.lock'
    sources['requirements-2022.lock'] = digest(lock)
    fact('NltkVersion', re.search(r'^nltk==(.+)$', lock.read_text(encoding='utf-8'), re.M).group(1), 'requirements-2022.lock', 'nltk version')
    fact('RealCheckCases', real_check['case_count'], 'real_check', 'case_count')
    fact('IterativeCases',len(iterative['cases']),'iterative','len(cases)')
    fact('IterativeSteps',iterative['iterations_per_case'],'iterative','iterations_per_case')
    fact('IterativeSeeds',iterative['seed_count'],'iterative','seed_count')
    fact('IterativeFreeFraction',iterative['cases'][0]['free_fraction'],'iterative','cases[0].free_fraction',2)
    coverage = next(x for x in revision_diagnostics['coverage'] if x['distribution'] == 'zipf_iid')
    for name, field, precision in [('ZipfObservedMin','minimum',None), ('ZipfObservedMax','maximum',None), ('ZipfObservedMean','mean',1)]:
        fact(name, coverage[field], 'revision_diagnostics', 'coverage.zipf_iid.' + field, precision)
    fact('IndependentKeywords', revision_diagnostics['independent_check']['keyword_count'], 'revision_diagnostics', 'independent_check.keyword_count')
    fact('IndependentDocuments', revision_diagnostics['independent_check']['documents'], 'revision_diagnostics', 'independent_check.documents')
    fact('NearDuplicateThreshold', revision_config['near_duplicate_jaccard'], 'revision_config', 'near_duplicate_jaccard', 2)
    fact('NearDuplicatePairs', sum(x['cross_partition_pair_count'] for x in revision_diagnostics['near_duplicates']), 'revision_diagnostics', 'sum(near_duplicates.cross_partition_pair_count)')
    fact('SensitivityPartitions', len(revision_config['partition_seeds']), 'revision_config', 'len(partition_seeds)')
    fact('SensitivitySeeds', len(revision_config['attack_seeds']), 'revision_config', 'len(attack_seeds)')
    fact('ControlledKeywords', revision_config['controlled_keywords'], 'revision_config', 'controlled_keywords')
    fact('SensitivityRuns', revision_results['run_count'], 'revision_results', 'run_count')
    fact('ProductionSplitSeed', production_split_config['seed'], 'production_split_config', 'seed')
    fact('MatchedSelectionSeed', matched_selection_config['seed'], 'matched_selection_config', 'seed')
    fact('BalancedSplitSeed', balanced_split_config['split_seed'], 'balanced_split_config', 'split_seed')
    fact('SensitivityPartitionSeeds', ', '.join(str(s) for s in revision_config['partition_seeds']), 'revision_config', 'partition_seeds')
    fact('SensitivityAttackSeeds', ', '.join(str(s) for s in revision_config['attack_seeds']), 'revision_config', 'attack_seeds')
    summaries = revision_results['summaries']
    lookup_revision = {(r['family'], r['partition'], r['language'], r['condition'], r['distribution']): r for r in summaries}
    original_named = [('stopword_removal','zipf_iid','weighted'), ('stemming','uniform_iid','weighted'),
                      ('stemming','uniform_iid','macro'), ('stemming','zipf_iid','weighted'), ('stemming','zipf_iid','macro')]
    differences = [lookup_revision[('partition', p, 'lus', c, d)][m] - lookup_revision[('partition', p, 'en', c, d)][m]
                   for p in revision_config['partition_seeds'] for c,d,m in original_named]
    fact('SensitivityPositiveDirections', sum(x > 0 for x in differences), 'revision_results', 'positive Mizo-minus-English means for the original named comparisons across additional partitions')
    fact('SensitivityDirectionCount', len(differences), 'revision_results', 'named comparisons times additional partitions')
    balanced_differences = [lookup_revision[('balanced', p, 'en', 'flores200', 'zipf_iid')]['weighted'] - lookup_revision[('balanced', p, 'en', 'wmt23', 'zipf_iid')]['weighted'] for p in revision_config['partition_seeds']]
    fact('BalancedSensitivityMin', min(balanced_differences), 'revision_results', 'minimum English Zipf weighted FLORES-minus-WMT23 partition mean', 4)
    fact('BalancedSensitivityMax', max(balanced_differences), 'revision_results', 'maximum English Zipf weighted FLORES-minus-WMT23 partition mean', 4)
    for language, prefix in [('en','EN'), ('lus','LUS')]:
        partition = next(r['partition'] for r in summaries if r['family']=='controlled')
        difference = lookup_revision[('controlled', partition, language, 'stemming', 'zipf_iid')]['weighted'] - lookup_revision[('controlled', partition, language, 'stopword_removal', 'zipf_iid')]['weighted']
        fact(prefix+'ControlledStemZipfDifference', difference, 'revision_results', language+' controlled Zipf weighted stemming-minus-stopwords mean', 4)
    shuffled = [r['weighted'] for r in summaries if r['family']=='shuffled']
    fact('ShuffledWeightedMin', min(shuffled), 'revision_results', 'minimum shuffled-rank weighted mean', 4)
    fact('ShuffledWeightedMax', max(shuffled), 'revision_results', 'maximum shuffled-rank weighted mean', 4)
    for name, value, source, key in [
        ('ProductionPairs', production['document_count'][0], 'production', 'document_count[0]'),
        ('Keywords', production['keyword_count'], 'production', 'keyword_count'),
        ('SeedCount', len(production['seeds']), 'production', 'len(seeds)'),
        ('ProductionRuns', production['run_count'], 'production', 'run_count'),
        ('LanguageSignificant', sum(x['holm_adjusted_p'] < 0.05 for x in production['language_comparisons']), 'production', 'language_comparisons adjusted p < 0.05'),
        ('PreprocessingSignificant', sum(x['holm_adjusted_p'] < 0.05 for x in production['preprocessing_comparisons']), 'production', 'preprocessing_comparisons adjusted p < 0.05'),
        ('BalancedPairs', balanced['matched_pair_count'], 'balanced', 'matched_pair_count'),
        ('BalancedSignificant', sum(x['holm_adjusted_p'] < 0.05 for x in balanced['comparisons']), 'balanced', 'comparisons adjusted p < 0.05'),
        ('MatchedPairs', matched['corpus_structure']['flores200']['paired_document_count'], 'matched', 'corpus_structure.flores200.paired_document_count'),
        ('SourceRows', audit['source_row_count'], 'audit', 'source_row_count'),
        ('SentencesPerBlock', wmt_config['sentences_per_document'], 'wmt_config', 'sentences_per_document'),
        ('EnglishStemMatched', stemming['by_language']['en']['correct_count'], 'stemming', 'by_language.en.correct_count'),
        ('EnglishStemTotal', stemming['by_language']['en']['item_count'], 'stemming', 'by_language.en.item_count'),
        ('MizoStemMatched', stemming['by_language']['lus']['correct_count'], 'stemming', 'by_language.lus.correct_count'),
        ('MizoStemTotal', stemming['by_language']['lus']['item_count'], 'stemming', 'by_language.lus.item_count'),
        ('UnderStemming', sum(value == 'Under-stemming' for value in clarification['mizo_error_types'].values()), 'clarification', 'under-stemming count'),
        ('OverStemming', sum(value == 'Over-stemming' for value in clarification['mizo_error_types'].values()), 'clarification', 'over-stemming count'),
        ('EnglishStopwords', review['stopwords']['en']['counts']['exclude_from_index'], 'review', 'stopwords.en.counts.exclude_from_index'),
        ('MizoStopwords', review['stopwords']['lus']['counts']['exclude_from_index'], 'review', 'stopwords.lus.counts.exclude_from_index'),
        ('MizoRules', len(rules['rules']), 'rules', 'len(rules)'),
        ('MinimumLetters', tokenisation['minimum_letter_count'], 'tokenisation', 'minimum_letter_count'),
        ('FinalIterations', production['checkpoint'], 'production', 'checkpoint'),
        ('QueriesPerKeyword', experiment['config']['query_count_per_keyword'], 'experiment', 'config.query_count_per_keyword'),
        ('LanguageFamily', len(production['language_comparisons']), 'production', 'len(language_comparisons)'),
        ('PreprocessingFamily', len(production['preprocessing_comparisons']), 'production', 'len(preprocessing_comparisons)'),
    ]:
        fact(name, value, source, key)
    fact('ConfidencePercent', production['config']['confidence_level'] * 100, 'production', 'config.confidence_level * 100', 0)
    fact('Alpha', 1 - production['config']['confidence_level'], 'production', '1 - config.confidence_level', 2)
    fact('LengthRatio', balanced['maximum_length_ratio'], 'balanced', 'maximum_length_ratio', 2)
    fact('FreeFraction', experiment['config']['free_fraction'], 'experiment', 'config.free_fraction', 2)
    keyword_config = interventions['keyword_size']
    fact('EnronClient', keyword_config['client_documents_after_split'], 'keyword_size', 'client_documents_after_split')
    fact('EnronAuxiliary', keyword_config['selected_documents'] - keyword_config['client_documents_after_split'], 'keyword_size', 'selected_documents - client_documents_after_split')
    fact('InterventionKeywords', interventions['access_collision']['keyword_count'], 'access_collision', 'keyword_count')
    fact('SimilarityPairs', interventions['access_similarity']['pair_count'], 'access_similarity', 'pair_count')
    fact('CollisionPairs', max(interventions['access_collision']['forced_pair_counts']), 'access_collision', 'max(forced_pair_counts)')
    fact('CandidateBudget', min(interventions['cooccurrence_similarity']['candidate_set_budgets']), 'cooccurrence_similarity', 'min(candidate_set_budgets)')
    for name, field in [('EnronMean', 'mean'), ('EnronLow', 'confidence_interval_low'), ('EnronHigh', 'confidence_interval_high')]:
        fact(name, enron['aggregate_metrics']['query_weighted_recovery']['1000'][field], 'enron', f'aggregate_metrics.query_weighted_recovery.1000.{field}', 4)
    fact('SmokeObserved', smoke['observed_accuracy'][-1], 'smoke', 'observed_accuracy[-1]', 3)
    fact('SmokeExpected', smoke['expected_recovery'][-1], 'smoke', 'expected_recovery[-1]', 3)
    for name, value in [('PythonVersion', platform.python_version()), ('NumpyVersion', np.__version__), ('ScipyVersion', scipy.__version__)]:
        fact(name, value, 'executing_runtime', name)
    fact('UpstreamRevision', experiment['upstream_revision'], 'experiment', 'upstream_revision')
    records = {r['dataset_id']: r for r in structure['records'] if r['manifest'] == 'results/corpus/ihop/manifest.json'}
    counts = {r['client']['identical_access_pair_count'] for r in records.values()}
    if len(counts) != 1:
        raise ValueError('Production identical-pair counts need a revised statement')
    fact('IdenticalPairs', counts.pop(), 'structure', 'common client.identical_access_pair_count')
    final = [r for r in production['aggregate'] if r['checkpoint'] == production['checkpoint']]
    index = {(r['language'], r['condition'], r['distribution'], r['metric']): r for r in final}
    for language, prefix in [('en', 'EN'), ('lus', 'LUS')]:
        for condition, short in [('tokenisation_only', 'Token'), ('stopword_removal', 'Stop'), ('stemming', 'Stem')]:
            r = index[(language, condition, 'uniform_iid', 'query_weighted_recovery')]
            fact(prefix + short + 'UniformPercent', r['mean'] * 100, 'production', f'{language}:{condition}:uniform_iid weighted mean * 100', 2)
        for metric, short in [('query_weighted_recovery', 'Weighted'), ('distinct_keyword_macro_recovery', 'Macro')]:
            r = index[(language, 'tokenisation_only', 'zipf_iid', metric)]
            fact(prefix + 'TokenZipf' + short, r['mean'], 'production', f'{language}:tokenisation_only:zipf_iid {metric} mean', 4)
    comparison = next(x for x in matched['comparisons'] if x['language'] == 'en' and x['query_distribution'] == 'zipf_iid' and x['metric'] == 'query_weighted_recovery')
    fact('MatchedEnglishDifference', comparison['mean_difference_flores_minus_wmt23'], 'matched', 'en zipf_iid weighted difference', 4)
    fact('MatchedEnglishP', comparison['holm_adjusted_p'], 'matched', 'en zipf_iid weighted Holm p', 4)
    for corpus, name in [('wmt23', 'WMT'), ('flores200', 'Flores')]:
        for language, suffix in [('en', 'EN'), ('lus', 'LUS')]:
            fact(name + 'Marker' + suffix, balanced['religious_markers'][corpus][language]['any_marker_document_percentage'], 'balanced', f'religious_markers.{corpus}.{language}.any_marker_document_percentage', 2)
    recovery_rows = []
    for language in ['en', 'lus']:
        for condition in CONDITIONS:
            for distribution in ['uniform_iid', 'zipf_iid']:
                weighted = index[(language, condition, distribution, 'query_weighted_recovery')]
                macro = index[(language, condition, distribution, 'distinct_keyword_macro_recovery')]
                recovery_rows.append([LANG[language], LABELS[condition], DIST[distribution],
                                      f"{weighted['mean']:.4f} ({weighted['sample_standard_deviation']:.4f})",
                                      f"{macro['mean']:.4f} ({macro['sample_standard_deviation']:.4f})"])
    blocks = {}
    blocks['RECOVERY_TABLE'] = table('Final recovery means and sample standard deviations.', 'tab:recovery', ['Language', 'Condition', 'Queries', 'Weighted (SD)', 'Macro (SD)'], recovery_rows, 'lllll')
    language_rows = [[LABELS[x['condition']], DIST[x['distribution']], METRIC[x['metric']], f"{x['mean']:.4f}", interval(x['confidence_interval']), f"{x['holm_adjusted_p']:.4f}"] for x in production['language_comparisons']]
    blocks['LANGUAGE_TABLE'] = table('Mizo minus English recovery at the final checkpoint.', 'tab:language', ['Condition', 'Queries', 'Metric', 'Difference', '95\% CI$^{a}$', 'Holm $p$'], language_rows, 'lllrrr')
    structure_rows = [[LANG[language], LABELS[condition], f"{records[f'{language}:{condition}']['client']['density']:.4f}", f"{records[f'{language}:{condition}']['client']['frequency_alternative_proportion']:.4f}", f"{records[f'{language}:{condition}']['client']['nearest_access_jaccard_median']:.4f}", f"{records[f'{language}:{condition}']['client']['cooccurrence_cosine_median']:.4f}"] for language in ['en', 'lus'] for condition in CONDITIONS]
    blocks['STRUCTURE_TABLE'] = table('Client-index density, frequency alternatives, and median similarities.', 'tab:structure', ['Language', 'Condition', 'Density', 'Alternatives', 'Nearest Jaccard', 'Cosine'], structure_rows, 'llrrrr')
    balanced_rows = [[LANG[x['language']], x['keyword_count'], f"{x['flores_mean']:.4f}", f"{x['wmt23_mean']:.4f}", f"{x['mean_difference_flores_minus_wmt23']:.4f}", interval(x['difference_95_percent_confidence_interval']), f"{x['holm_adjusted_p']:.4f}"] for x in balanced['comparisons'] if x['query_distribution'] == 'zipf_iid' and x['metric'] == 'query_weighted_recovery']
    blocks['BALANCED_TABLE'] = table('Selected Zipf query-weighted recovery contrasts after length balancing.', 'tab:balanced', ['Language', '$K$', 'FLORES', 'WMT23', 'Difference', '95\% CI$^{a}$', 'Holm $p$'], balanced_rows, 'lrrrrrr')
    for block_name in ['LANGUAGE_TABLE', 'BALANCED_TABLE']:
        blocks[block_name] = re.sub(r'\[(-?\d+\.\d+), (-?\d+\.\d+)\]', lambda m:r'\shortstack{['+m.group(1)+r',\\'+m.group(2)+']}', blocks[block_name])
        blocks[block_name] = blocks[block_name].replace(r'\end{tabular}', r'\end{tabular}\par\footnotesize $^{a}$Unadjusted confidence intervals.')
    ablation_rows = []
    for x in ablation['contrasts']:
        d = x['checkpoints']['1000']['paired_difference']
        ablation_rows.append([escape(x['label']), escape(x['contrast_label']), f"{d['mean']:.4f}", interval([d['confidence_interval_low'], d['confidence_interval_high']])])
    blocks['ABLATION_TABLE'] = table('Paired final-recovery differences at the configured Enron intervention endpoints.', 'tab:ablation', ['Intervention', 'Endpoint contrast', 'Difference', '95\% CI'], ablation_rows, 'p{29mm}p{53mm}rr')
    matched_manifest = load('results/corpus/flores200-ihop/manifest.json', sources)
    entry = next(iter(matched_manifest['datasets'].values()))
    balanced_manifest = load('results/corpus/flores200-length-balanced-ihop/manifest.json', sources)
    balanced_entry = next(iter(balanced_manifest['datasets'].values()))
    corpus_rows = [['WMT23 production', production['document_count'][0], production['auxiliary_count'][0], production['client_count'][0]],
                   ['FLORES and WMT23 matched', facts['MatchedPairs']['value'], entry['auxiliary_document_count'], entry['client_document_count']],
                   ['FLORES and WMT23 length-balanced', balanced['matched_pair_count'], balanced_entry['auxiliary_document_count'], balanced_entry['client_document_count']]]
    blocks['CORPUS_TABLE'] = table('Paired collection sizes and fixed partition sizes per language.', 'tab:corpora', ['Collection', 'Pairs', 'Auxiliary', 'Client'], corpus_rows, 'lrrr')
    blocks['RECOVERY_FIGURE'] = recovery_figure(production)
    blocks['PARTITION_FIGURE'] = figure_tex(uncertainty)
    blocks['FACTS'] = '\n'.join(f"\\newcommand{{\\{name}}}{{{escape(fact['rendered'])}}}" for name, fact in facts.items())
    revision = facts['UpstreamRevision']['rendered']
    blocks['COMMIT_ID'] = r'\texttt{' + r'\allowbreak{}'.join(revision[index:index+4] for index in range(0, len(revision), 4)) + '}'
    blocks['REFERENCES'] = '\n'.join([r'\begin{thebibliography}{99}'] + [f'\\bibitem{{{key}}} {escape(authors)}. {escape(title)}. {escape(venue)}. \\url{{{url}}}.' for key, authors, title, venue, url in REFERENCES] + [r'\end{thebibliography}'])
    template = ROOT / 'scripts/manuscript_template.tex'
    submission = load('paper/submission-information.json', sources)
    sources['scripts/manuscript_template.tex'] = digest(template)
    tex = template.read_text(encoding='utf-8')
    for key, value in blocks.items():
        tex = tex.replace('@@' + key + '@@', value)
    if '@@' in tex:
        raise ValueError('Unresolved manuscript content marker')
    citations = {key for group in re.findall(r'\\cite\{([^}]+)\}', tex) for key in group.split(',')}
    if citations != {r[0] for r in REFERENCES}:
        raise ValueError('Manuscript reference keys are missing or unused')
    order = list(dict.fromkeys(key for group in re.findall(r'\\cite\{([^}]+)\}',tex) for key in group.split(',')))
    reference_map = {r[0]:r for r in REFERENCES}
    ordered_references = [reference_map[key] for key in order]
    bibliography = '\n'.join([r'\begin{thebibliography}{99}']+[f'\\bibitem{{{key}}} {escape(authors)}. {escape(title)}. {escape(venue)}. \\url{{{url}}}.' for key,authors,title,venue,url in ordered_references]+[r'\end{thebibliography}'])
    tex = tex.replace(blocks['REFERENCES'],bibliography)
    supplement = supplementary(production, matched, sensitivity, balanced, stemming, clarification, blocks['FACTS'], sources,
                               rules, revision_diagnostics, revision_controls, revision_results, experiment, uncertainty)
    for name in ['sn-article.tex', 'sn-jnl.cls', 'bst/sn-mathphys-num.bst']:
        path = OFFICIAL / name
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    tex, supplement = migrate(tex, supplement, submission['authors'])
    ledger = {'sources': sources, 'facts': facts, 'references': [dict(key=k, authors=a, title=t, venue=v, url=u) for k,a,t,v,u in ordered_references],
              'table_rows': {'recovery': recovery_rows, 'language': language_rows, 'structure': structure_rows, 'balanced_selected': balanced_rows, 'enron_endpoints': ablation_rows, 'corpora': corpus_rows},
              'author_policy': 'The six-author order, affiliations and email addresses were supplied by the user.',
              'authors': submission['authors'], 'corresponding_author_email': submission['corresponding_author_email']}
    return tex, supplement, ledger


def supplementary(production, matched, sensitivity, balanced, stemming, clarification, facts, sources,
                  rules, revision_diagnostics, revision_controls, revision_results, experiment, uncertainty):
    parts = [r'\documentclass[12pt]{article}', r'\usepackage[a4paper,landscape,margin=20mm]{geometry}', r'\usepackage{longtable,booktabs,array,hyperref,helvet}',
             r'\renewcommand{\familydefault}{\sfdefault}', r'\renewcommand{\small}{\normalsize}', r'\renewcommand{\footnotesize}{\normalsize}',
             r'\hypersetup{hidelinks,pdfauthor={}}', r'\setlength{\tabcolsep}{3pt}', facts,
             r'\begin{document}', r'\renewcommand{\thetable}{S\arabic{table}}', r'\section*{Supplementary material}',
             'Full recovery comparisons and fixed-sample stemming disagreements were retained below. Differences were computed within paired seeds. All recovery measurements were proportions. Confidence intervals were unadjusted. Holm probabilities used the comparison families specified in Methods. Corpus-choice controls used tokenisation only.',
             r'Tables~\ref{tab:supp:1}, \ref{tab:supp:2}, \ref{tab:supp:3}, and \ref{tab:supp:4} show recovery comparisons. Table~\ref{tab:supp:5} shows stemming disagreements. Table~\ref{tab:supp:6} shows illustrative worked examples.',
             r'\small']
    table_number = 0
    def long_table(caption, headers, rows, cols):
        nonlocal table_number
        table_number += 1
        parts.extend([f'\\begin{{longtable}}{{@{{}}{cols}@{{}}}}', f'\\caption{{{caption}}}\\label{{tab:supp:{table_number}}}\\\\', r'\toprule', ' & '.join(headers) + r' \\', r'\midrule\endfirsthead', r'\toprule', ' & '.join(headers) + r' \\', r'\midrule\endhead'])
        parts.extend(' & '.join(map(str, row)) + r' \\' for row in rows)
        parts.extend([r'\bottomrule\end{longtable}'])
    rows = [[LANG[x['language']], LABELS[x['left']], LABELS[x['right']], DIST[x['distribution']], METRIC[x['metric']], f"{x['mean']:.4f}", interval(x['confidence_interval']), f"{x['holm_adjusted_p']:.4f}"] for x in production['preprocessing_comparisons']]
    long_table('Production preprocessing differences, condition minus reference.', ['Language','Condition','Reference','Queries','Metric','Diff.','95\% CI','Holm $p$'], rows, 'lllllrrr')
    for name, result in [('Matched-size corpus comparisons', matched), ('Keyword-size corpus comparisons', sensitivity), ('Length-balanced corpus comparisons', balanced)]:
        comparisons = result.get('comparisons', result.get('sensitivity_comparisons'))
        rows = [[LANG[x['language']], x.get('keyword_count', production['keyword_count']), DIST[x['query_distribution']], METRIC[x['metric']], f"{x['flores_mean']:.4f}", f"{x['wmt23_mean']:.4f}", f"{x['mean_difference_flores_minus_wmt23']:.4f}", f"{x['holm_adjusted_p']:.4f}"] for x in comparisons]
        long_table(name + ', FLORES-200 minus WMT23.', ['Language','$K$','Queries','Metric','FLORES','WMT23','Diff.','Holm $p$'], rows, 'lrllrrrr')
    errors = clarification['mizo_error_types']
    disagreements = [x for x in stemming['items'] if not x['correct']]
    if {x['token'] for x in disagreements if x['language'] == 'lus'} != set(errors):
        raise ValueError('Mizo error classification differs from measured disagreements')
    rows = [[LANG[x['language']], escape(x['token']), escape(x['predicted_stem']), escape(x['expected_stem']), errors.get(x['token'], 'Mismatch')] for x in disagreements]
    long_table('Stemming disagreements in the fixed reviewed annotations.', ['Language','Token','Output','Expected','Error'], rows, 'lllll')
    parts.append(r'Mizo disagreements comprised \UnderStemming{} under-stemming cases and \OverStemming{} over-stemming cases. Error categories followed the reviewer clarification.')
    rows = [[escape(x['token']), escape(x['output']), escape(x['expected'])] for x in stemming['worked_examples']]
    long_table('Illustrative worked examples of the clarified Mizo procedure.', ['Token', 'Output', 'Expected'], rows, 'lll')
    parts.append('Stemming agreement was measured on the reviewed development sample.')
    parts.append(r'Table~\ref{tab:supp:7} specifies the accepted Mizo suffix rules. Table~\ref{tab:supp:8} lists the excluded stopwords. Table~\ref{tab:supp:9} reports vocabulary changes. Table~\ref{tab:supp:10} reports query coverage. Tables~\ref{tab:supp:11}, \ref{tab:supp:12}, and \ref{tab:supp:13} report the post-review sensitivity results.')
    rows = [[escape(r['suffix']), r['priority'], r['minimum_stem_letter_count'], escape(r['replacement']) or '(empty)', escape(', '.join(r['exceptions'])) or '(none)'] for r in sorted(rules['rules'], key=lambda r:(r['priority'], -len(r['suffix']), r['suffix']))]
    long_table('Accepted Mizo suffix rules and current-form exceptions.', ['Suffix', 'Priority', 'Minimum letters', 'Replacement', 'Exceptions'], rows, 'lllp{20mm}p{80mm}')
    parts.append('Rules were tried in ascending priority and decreasing suffix length within a priority. The first eligible rule was applied. The procedure was repeated on the current form until no rule applied. Exceptions blocked their associated rule. NFC-normalised written letters were counted once. Corpus attestation was excluded from eligibility.')
    rows = []
    for language in ['en', 'lus']:
        stopwords = load(f'resources/stopwords/{language}-production.reviewed.json', sources)
        terms = sorted(x['token'] for x in stopwords['entries'] if x['decision'] == 'exclude_from_index')
        rows.append([LANG[language], escape(', '.join(terms))])
    long_table('Reviewed stopwords excluded from each index.', ['Language', 'Tokens'], rows, 'lp{143mm}')
    rows = [[LANG[x['language']], x['token_stop_overlap'], x['stop_stem_string_overlap'], x['stop_types'], x['stem_types'], x['merged_stem_groups'], x['types_in_merged_groups']] for x in revision_diagnostics['vocabulary']]
    long_table('Selected-keyword overlap and full-vocabulary stem mergers.', ['Language', 'Token/stop', 'Stop/stem', 'Stop types', 'Stem types', 'Merged groups', 'Merged types'], rows, 'lrrrrrr')
    parts.append('Overlap counted identical written labels in the selected production keyword sets. Merger counts used the complete language-specific stopword-filtered vocabulary. A merged group contained multiple word types with the same output stem.')
    rows = [[DIST[x['distribution']], x['minimum'], x['maximum'], f"{x['mean']:.1f}"] for x in revision_diagnostics['coverage']]
    long_table('Observed distinct keywords in production queries.', ['Queries', 'Minimum', 'Maximum', 'Mean'], rows, 'lrrr')
    def measurement(x, metric):
        return f"{x[metric]['mean']:.4f} ({x[metric]['sd']:.4f})"
    rows = [[x['partition'], LANG[x['language']], LABELS[x['condition']], DIST[x['distribution']], measurement(x, 'weighted'), measurement(x, 'macro')] for x in uncertainty['summaries'] if x['family'] == 'partition']
    long_table('Production recovery means and sample standard deviations within additional partitions.', ['Partition seed', 'Language', 'Condition', 'Queries', 'Weighted (SD)', 'Macro (SD)'], rows, 'rlllrr')
    rows = [[('Controlled' if x['family']=='controlled' else 'Shuffled ranks'), LANG[x['language']], LABELS[x['condition']], DIST[x['distribution']], measurement(x,'weighted'), measurement(x,'macro')] for x in uncertainty['summaries'] if x['family'] in ['controlled', 'shuffled']]
    for language in ['en', 'lus']:
        for condition in CONDITIONS:
            values = [r for r in experiment['runs'] if r['dataset_id']==f'{language}:{condition}' and r['distribution']=='zipf_iid' and r['seed'] in revision_results['config']['attack_seeds']]
            rows.append(['Frequency ranks', LANG[language], LABELS[condition], 'Zipf',
                         f"{np.mean([r['scores'][-1]['query_weighted_recovery'] for r in values]):.4f} ({np.std([r['scores'][-1]['query_weighted_recovery'] for r in values], ddof=1):.4f})",
                         f"{np.mean([r['scores'][-1]['distinct_keyword_macro_recovery'] for r in values]):.4f} ({np.std([r['scores'][-1]['distinct_keyword_macro_recovery'] for r in values], ddof=1):.4f})"])
    long_table('Recovery means and sample standard deviations for keyword and rank controls.', ['Control', 'Language', 'Condition', 'Queries', 'Weighted (SD)', 'Macro (SD)'], rows, 'llllrr')
    rows = [[x['partition'], LANG[x['language']], ('FLORES' if x['condition']=='flores200' else 'WMT23'), DIST[x['distribution']], measurement(x,'weighted'), measurement(x,'macro')] for x in uncertainty['summaries'] if x['family'] == 'balanced']
    long_table('Length-balanced recovery means and sample standard deviations within additional partitions.', ['Partition seed', 'Language', 'Corpus', 'Queries', 'Weighted (SD)', 'Macro (SD)'], rows, 'rlllrr')
    parts.append(r'Table~\ref{tab:supp:14} reports paired contrast means and sample standard deviations within additional partitions.')
    rows = [[x['partition'], ('Language' if x['family']=='partition' else LANG[x['language']]+' corpus'), LABELS.get(x['condition'],'Tokenisation'), DIST[x['distribution']], x['metric'].title(), f"{x['mean']:.4f}", f"{x['sd']:.4f}"] for x in uncertainty['contrasts']]
    long_table('Paired language and length-balanced corpus contrasts within additional partitions.', ['Partition seed','Contrast','Condition','Queries','Metric','Mean','SD'], rows, 'rllllrr')
    parts.append('Language contrasts were Mizo minus English. Corpus contrasts were FLORES minus WMT23 at 500 keywords. Standard deviations used seed-wise paired differences. Seed-level values were retained in the derived uncertainty results. These standard deviations described within-partition variation.')
    parts.append(r'Table~\ref{tab:supp:15} reports original partition contrasts using the same attack seeds as the additional partitions.')
    rows = [[('Language' if x['family']=='partition' else LANG[x['language']]+' corpus'), LABELS[x['condition']], DIST[x['distribution']], x['metric'].title(), f"{x['mean']:.4f}", f"{x['sd']:.4f}"] for x in uncertainty['matched_seed_baseline']]
    long_table('Original partition paired contrasts on matched attack seeds.', ['Contrast','Condition','Queries','Metric','Mean','SD'], rows, 'llllrr')
    parts.append(r'Each post-review mean used \SensitivitySeeds{} attack seeds. These tables were descriptive. The confirmatory comparison families were preserved. Controlled keyword labels were linked through the frozen stemmer. Each selected word had a distinct output stem. The tokenisation-only and stopword-only restricted matrices were identical. Full controlled mappings were retained in the accompanying derived results.')
    parts.extend([r'\section*{Provenance}', 'SHA-256 identifiers for the source results and configurations were recorded below. Source text was excluded.'])
    for name, checksum in sources.items():
        parts.extend([f'\\noindent\\texttt{{{escape(name)}}}\\\\', f'\\texttt{{{checksum}}}\\par'])
    parts.append(r'\end{document}')
    return '\n'.join(parts) + '\n'


def main():
    tex, supplement, ledger = build()
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / 'manuscript.tex').write_text(tex, encoding='utf-8')
    (OUTPUT / 'supplement.tex').write_text(supplement, encoding='utf-8')
    (OUTPUT / 'numerical-provenance.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    (OUTPUT / 'sn-bibliography.bib').write_text(bibliography(ledger['references']), encoding='utf-8')
    print(f"Built manuscript, supplementary tables, and {len(ledger['facts'])} reproducible facts")


if __name__ == '__main__':
    main()
