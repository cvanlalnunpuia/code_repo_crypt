from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy import stats

from analyse_matched_domain_comparison import exact_sign_flip_p, file_sha256, holm_adjust


LABELS = {'tokenisation_only': 'Tokenisation', 'stopword_removal': 'Stopwords', 'stemming': 'Stemming'}
LANGUAGES = {'en': 'English', 'lus': 'Mizo'}
DISTRIBUTIONS = {'uniform_iid': 'Uniform IID', 'zipf_iid': 'Zipf IID'}
METRICS = {'query_weighted_recovery': 'Query weighted', 'distinct_keyword_macro_recovery': 'Keyword macro'}


def summarise(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    margin = stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values))
    return {'count': len(values), 'mean': mean, 'sample_standard_deviation': sd,
            'confidence_interval': [mean - margin, mean + margin]}


def paired(left, right):
    if len(left) != len(right) or len(left) < 2:
        raise ValueError('Paired samples must have equal lengths and at least two values')
    differences = [a - b for a, b in zip(left, right)]
    result = summarise(differences)
    result.update(left_mean=statistics.mean(left), right_mean=statistics.mean(right),
                  differences=differences, exact_sign_flip_p=exact_sign_flip_p(differences))
    return result


def build(root):
    config_path = root / 'configs/production_recovery_analysis.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['confidence_level'] != 0.95:
        raise ValueError('Unsupported confidence level')
    path = root / config['experiment']
    experiment = json.loads(path.read_text(encoding='utf-8'))
    manifest_path = root / experiment['config']['manifest']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if experiment['manifest_sha256'] != file_sha256(manifest_path):
        raise ValueError('Experiment manifest changed')
    seeds = experiment['config']['seeds']
    distributions = experiment['config']['query_distributions']
    checkpoints = experiment['config']['iteration_checkpoints']
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError('Seeds must be unique and contain at least two values')
    index = {(r['dataset_id'], r['distribution'], r['seed']): r for r in experiment['runs']}
    expected = {(f'{language}:{condition}', distribution, seed) for language in config['languages']
                for condition in config['conditions'] for distribution in distributions for seed in seeds}
    if len(index) != len(experiment['runs']) or set(index) != expected:
        raise ValueError('Production run keys are duplicated or incomplete')
    for run in experiment['runs']:
        item = manifest['datasets'][run['dataset_id']]
        if run['checkpoints'] != checkpoints or run['dataset_sha256'] != item['sha256']:
            raise ValueError('Run provenance or checkpoints differ')
        if run['keyword_count'] != manifest['config']['keyword_universe_size']:
            raise ValueError('Keyword count differs from specification')
        if len(run['scores']) != len(checkpoints):
            raise ValueError('Score count differs from checkpoint count')
        for scores in run['scores']:
            for metric in config['metrics']:
                if not math.isfinite(scores[metric]) or not 0 <= scores[metric] <= 1:
                    raise ValueError('Recovery score must be finite and within zero and one')
    def values(language, condition, distribution, metric, checkpoint):
        position = checkpoints.index(checkpoint)
        return [index[(f'{language}:{condition}', distribution, seed)]['scores'][position][metric] for seed in seeds]
    aggregates = []
    for language in config['languages']:
        for condition in config['conditions']:
            for distribution in distributions:
                for checkpoint in checkpoints:
                    for metric in config['metrics']:
                        aggregate = summarise(values(language, condition, distribution, metric, checkpoint))
                        aggregate.update(language=language, condition=condition, distribution=distribution,
                                         checkpoint=checkpoint, metric=metric)
                        aggregates.append(aggregate)
    final = max(checkpoints)
    language_comparisons = []
    for condition in config['conditions']:
        for distribution in distributions:
            for metric in config['metrics']:
                item = paired(values('lus', condition, distribution, metric, final), values('en', condition, distribution, metric, final))
                item.update(condition=condition, distribution=distribution, metric=metric)
                language_comparisons.append(item)
    preprocessing_comparisons = []
    for language in config['languages']:
        for left, right in config['preprocessing_contrasts']:
            for distribution in distributions:
                for metric in config['metrics']:
                    item = paired(values(language, left, distribution, metric, final), values(language, right, distribution, metric, final))
                    item.update(language=language, left=left, right=right, distribution=distribution, metric=metric)
                    preprocessing_comparisons.append(item)
    holm_adjust(language_comparisons)
    holm_adjust(preprocessing_comparisons)
    return {'config': config, 'run_count': len(index), 'seeds': seeds, 'checkpoint': final,
            'query_count_per_run': sorted({run['query_count'] for run in experiment['runs']}),
            'keyword_count': manifest['config']['keyword_universe_size'],
            'document_count': sorted({item['document_count'] for item in manifest['datasets'].values()}),
            'auxiliary_count': sorted({item['auxiliary_document_count'] for item in manifest['datasets'].values()}),
            'client_count': sorted({item['client_document_count'] for item in manifest['datasets'].values()}),
            'aggregate': aggregates, 'language_comparisons': language_comparisons,
            'preprocessing_comparisons': preprocessing_comparisons,
            'inputs': {'experiment_sha256': file_sha256(path), 'analysis_config_sha256': file_sha256(config_path),
                       'manifest_sha256': file_sha256(manifest_path)}}


def interval(values):
    return f'[{values[0]:.4f}, {values[1]:.4f}]'


def markdown(result):
    lines = ['# Recovery', '',
             f"The production experiment executed {result['run_count']} runs over {len(result['seeds'])} seeds. Each dataset used {result['keyword_count']} selected keywords and {result['query_count_per_run'][0]} query observations. The fixed split contained {result['auxiliary_count'][0]} auxiliary and {result['client_count'][0]} client documents per language.", '',
             f"Table 1 shows final recovery after {result['checkpoint']} iterations of the Iterative Heuristic Optimization attack (IHOP). Uniform and Zipf query observations were independent and identically distributed (IID). Query-weighted recovery counted every observed query. Keyword-macro recovery averaged recovery over the observed distinct keywords. Each mean was reported with its standard deviation (SD) and 95% confidence interval (CI).", '',
             'The attack used volume leakage with no defence. The stemming condition included reviewed stopword removal. The Mizo suffix rules and execution checks were recorded in [the stemming evaluation](stemmer-execution.md).', '',
             'Table 1. Final recovery by language, preprocessing condition, and query distribution.', '',
             '| Language | Condition | Queries | Metric | Mean | SD | 95% CI |', '|---|---|---|---|---:|---:|---|']
    for item in result['aggregate']:
        if item['checkpoint'] == result['checkpoint']:
            lines.append(f"| {LANGUAGES[item['language']]} | {LABELS[item['condition']]} | {DISTRIBUTIONS[item['distribution']]} | {METRICS[item['metric']]} | {item['mean']:.4f} | {item['sample_standard_deviation']:.4f} | {interval(item['confidence_interval'])} |")
    lines.extend(['', 'Figure 1 shows mean query-weighted recovery across iteration checkpoints.', '',
                  '![Recovery across iteration checkpoints](../figures/production-recovery.png)', '',
                  'Figure 1. Mean query-weighted recovery across iterations. Panels show language and query distribution. Colours show preprocessing conditions.', '',
                  f"Table 2 shows Mizo minus English recovery. Holm adjustment covered {len(result['language_comparisons'])} language comparisons.", '',
                  'Table 2. Paired language differences at the final checkpoint.', '',
                  '| Condition | Queries | Metric | Difference | 95% CI | Exact p | Holm p |', '|---|---|---|---:|---|---:|---:|'])
    for item in result['language_comparisons']:
        lines.append(f"| {LABELS[item['condition']]} | {DISTRIBUTIONS[item['distribution']]} | {METRICS[item['metric']]} | {item['mean']:.4f} | {interval(item['confidence_interval'])} | {item['exact_sign_flip_p']:.4f} | {item['holm_adjusted_p']:.4f} |")
    count = sum(item['holm_adjusted_p'] < 0.05 for item in result['language_comparisons'])
    lines.extend(['', f'{count} language comparisons had Holm-adjusted p below 0.05.', '',
                  f"Table 3 shows paired preprocessing contrasts. Differences are the first condition minus the reference condition. Holm adjustment covered {len(result['preprocessing_comparisons'])} preprocessing comparisons.", '',
                  'Table 3. Paired recovery differences between preprocessing conditions.', '',
                  '| Language | Condition | Reference | Queries | Metric | Difference | 95% CI | Holm p |', '|---|---|---|---|---|---:|---|---:|'])
    for item in result['preprocessing_comparisons']:
        lines.append(f"| {LANGUAGES[item['language']]} | {LABELS[item['left']]} | {LABELS[item['right']]} | {DISTRIBUTIONS[item['distribution']]} | {METRICS[item['metric']]} | {item['mean']:.4f} | {interval(item['confidence_interval'])} | {item['holm_adjusted_p']:.4f} |")
    count = sum(item['holm_adjusted_p'] < 0.05 for item in result['preprocessing_comparisons'])
    lines.extend(['', f'{count} preprocessing comparisons had Holm-adjusted p below 0.05.', '',
                  'Recovery intervals described variation across seeds. Difference intervals used paired seed differences. Student t intervals were used. Exact sign-flip tests evaluated all sign assignments of the paired differences. The two comparison families were specified separately in the analysis configuration before recovery results were available.', '',
                  'Preprocessing changed the indexed representation and the selected keyword vocabulary. The contrasts therefore measured the complete preprocessing and keyword-selection pipeline. Document pairs and split membership remained fixed.', '',
                  'The language comparisons were conditional on this corpus and split. Document construction was recorded in [the WMT23 corpus audit](wmt23-corpus-audit.md). The Mizo stemming implementation used the clarified repeated suffix procedure and retained documented annotation disagreements. Corpus occurrence was excluded from suffix eligibility.', '',
                  'All reported values were generated by scripts/analyse_production_recovery.py.', ''])
    return '\n'.join(lines)


def figure(root, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'ps.fonttype':42,'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.4), sharex=True, sharey=True)
    colors = {'tokenisation_only': '#245A81', 'stopword_removal': '#B46A20', 'stemming': '#26735F'}
    for row, language in enumerate(['en', 'lus']):
        for col, distribution in enumerate(['uniform_iid', 'zipf_iid']):
            ax = axes[row, col]
            for condition in result['config']['conditions']:
                points = sorted([item for item in result['aggregate'] if item['language'] == language and item['distribution'] == distribution and item['condition'] == condition and item['metric'] == 'query_weighted_recovery'], key=lambda item: item['checkpoint'])
                ax.plot([p['checkpoint'] + 1 for p in points], [p['mean'] for p in points], color=colors[condition], marker={'tokenisation_only':'o','stopword_removal':'s','stemming':'^'}[condition], markersize=4, linewidth=1.5, label=LABELS[condition])
            ax.set_xscale('log')
            ax.set_xticks([1, 11, 101, 1001], ['0', '10', '100', '1000'])
            ax.set_ylim(0, 1.03)
            ax.text(0,1.03,f'({"abcd"[row*2+col]})',transform=ax.transAxes)
            ax.grid(axis='y', color='#D8DEE3', linewidth=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if col == 0:
                ax.set_ylabel('Query-weighted recovery')
            if row == 1:
                ax.set_xlabel('IHOP iterations')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.16, hspace=0.32, wspace=0.16)
    (root / 'figures').mkdir(exist_ok=True)
    fig.savefig(root / 'figures/production-recovery.png', dpi=180)
    fig.savefig(root / 'figures/Fig1.eps')
    fig.savefig(root / 'paper/Fig1.pdf')
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    result = build(root)
    (root / 'results/corpus/production-recovery-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (root / 'docs/production-recovery.md').write_text(markdown(result), encoding='utf-8')
    figure(root, result)
    print(f"Analysed {result['run_count']} production runs with {len(result['language_comparisons'])} language and {len(result['preprocessing_comparisons'])} preprocessing contrasts")


if __name__ == '__main__':
    main()
