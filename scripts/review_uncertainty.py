from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/corpus/reviewer-sensitivity'
METRICS = [('weighted', 'query_weighted_recovery'), ('macro', 'distinct_keyword_macro_recovery')]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def summary(values):
    return {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1))}


def build():
    checkpoint = read('results/corpus/reviewer-sensitivity/checkpoint.json')
    analysis = read('results/corpus/reviewer-sensitivity/analysis.json')
    groups = defaultdict(list)
    for run in checkpoint['runs'].values():
        task = run['task']
        groups[tuple(task[k] for k in ['family', 'partition', 'language', 'condition', 'distribution'])].append(run)
    rows = []
    lookup = {}
    for group in analysis['summaries']:
        key = tuple(group[k] for k in ['family', 'partition', 'language', 'condition', 'distribution'])
        runs = sorted(groups[key], key=lambda r: r['seed'])
        if [r['seed'] for r in runs] != analysis['config']['attack_seeds']:
            raise ValueError('Sensitivity seed set differs')
        lookup[key] = {r['seed']: r['scores'][-1] for r in runs}
        row = {k: group[k] for k in ['family', 'partition', 'language', 'condition', 'distribution']}
        for name, metric in METRICS:
            row[name] = summary([r['scores'][-1][metric] for r in runs])
            if row[name]['mean'] != group[name]:
                raise ValueError('Sensitivity mean differs from saved analysis')
        rows.append(row)
    contrasts = []
    for key in sorted(lookup):
        family, partition, language, condition, distribution = key
        if family == 'partition' and language == 'lus':
            right = ('partition', partition, 'en', condition, distribution)
            contrast = 'Mizo minus English'
        elif family == 'balanced' and condition == 'flores200':
            right = ('balanced', partition, language, 'wmt23', distribution)
            contrast = 'FLORES minus WMT23'
        else:
            continue
        for name, metric in METRICS:
            values = [lookup[key][s][metric] - lookup[right][s][metric] for s in analysis['config']['attack_seeds']]
            contrasts.append(dict(family=family, partition=partition, language=language,
                                  condition=condition, distribution=distribution, metric=name,
                                  contrast=contrast, paired_values=values, **summary(values)))
    baseline = []
    production = read('results/corpus/production-fixed-split-experiments.json')['runs']
    seeds = analysis['config']['attack_seeds']
    def values(records, dataset, distribution, metric):
        return {r['seed']:r['scores'][-1][metric] for r in records if r['dataset_id']==dataset and r['distribution']==distribution and r['seed'] in seeds}
    for condition in ['tokenisation_only','stopword_removal','stemming']:
        for distribution in ['uniform_iid','zipf_iid']:
            for name, metric in METRICS:
                left, right = values(production,'lus:'+condition,distribution,metric), values(production,'en:'+condition,distribution,metric)
                paired = [left[s]-right[s] for s in seeds]
                baseline.append(dict(family='partition', condition=condition, distribution=distribution, metric=name, contrast='Mizo minus English', paired_values=paired, **summary(paired)))
    flores = read('results/corpus/flores200-length-balanced-experiments.json')['runs']
    wmt = read('results/corpus/wmt23-length-balanced-experiments.json')['runs']
    for language in ['en','lus']:
        for distribution in ['uniform_iid','zipf_iid']:
            for name, metric in METRICS:
                dataset = language+':tokenisation_only:k500'
                left, right = values(flores,dataset,distribution,metric), values(wmt,dataset,distribution,metric)
                paired = [left[s]-right[s] for s in seeds]
                baseline.append(dict(family='balanced', language=language, condition='tokenisation_only', distribution=distribution, metric=name, contrast='FLORES minus WMT23', paired_values=paired, **summary(paired)))
    return {'checkpoint_sha256': hashlib.sha256((OUT / 'checkpoint.json').read_bytes()).hexdigest(),
            'summaries': rows, 'contrasts': contrasts, 'matched_seed_baseline':baseline,
            'interpretation': 'Sample standard deviations described seed variation within fixed partitions. No additional confirmatory tests were performed.'}


def panels(result):
    production = read('results/corpus/production-recovery-analysis.json')
    balanced = read('results/corpus/length-balanced-comparison.json')
    names = [('stopword_removal', 'zipf_iid', 'weighted', 'Stopwords, Zipf weighted'),
             ('stemming', 'uniform_iid', 'weighted', 'Stemming, uniform weighted'),
             ('stemming', 'uniform_iid', 'macro', 'Stemming, uniform macro'),
             ('stemming', 'zipf_iid', 'weighted', 'Stemming, Zipf weighted'),
             ('stemming', 'zipf_iid', 'macro', 'Stemming, Zipf macro')]
    left = []
    for condition, distribution, metric, label in names:
        original = next(r for r in result['matched_seed_baseline'] if r['family']=='partition' and r['condition']==condition and r['distribution']==distribution and r['metric']==metric)
        values = [('Original', original['mean'])]
        values.extend((str(r['partition']), r['mean']) for r in result['contrasts'] if r['family'] == 'partition' and r['condition'] == condition and r['distribution'] == distribution and r['metric'] == metric)
        left.append((label, values))
    original = next(r for r in result['matched_seed_baseline'] if r['family']=='balanced' and r['language']=='en' and r['distribution']=='zipf_iid' and r['metric']=='weighted')
    right = [('Original', original['mean'])]
    right.extend((str(r['partition']), r['mean']) for r in result['contrasts'] if r['family'] == 'balanced' and r['language'] == 'en' and r['distribution'] == 'zipf_iid' and r['metric'] == 'weighted')
    return left, right


def figure_tex(result):
    left, right = panels(result)
    colours = ['black', 'blue', 'orange', 'teal']
    lines = [r'\begin{figure}[htbp]', r'\centering', r'\setlength{\unitlength}{0.95pt}', r'\begin{picture}(470,220)',
             r'\put(130,205){\makebox(0,0)[l]{\scriptsize (a)}}',
             r'\put(330,205){\makebox(0,0)[l]{\scriptsize (b)}}']
    for lo, width in [(130,150),(330,130)]:
        x = lambda v: lo + width * (v + .15) / .35
        lines.append(f'\\put({x(0):.2f},40){{\\line(0,1){{145}}}}')
        lines.append(f'\\put({lo},40){{\\line(1,0){{{width}}}}}')
        for tick in [-.1, 0, .1, .2]:
            lines.append(f'\\put({x(tick):.2f},29){{\\makebox(0,0){{\\scriptsize {tick:g}}}}}')
    for i, (label, values) in enumerate(left):
        y = 176 - 28*i
        lines.append(f'\\put(125,{y}){{\\makebox(0,0)[r]{{\\scriptsize {label}}}}}')
        for j, (_, value) in enumerate(values):
            x = 130 + 150 * (value+.15)/.35
            symbol = ['\\bullet','\\square','\\triangle','\\diamond'][j]
            lines.append(f'{{\\color{{{colours[j]}}}\\put({x:.2f},{y+6-4*j}){{\\makebox(0,0){{\\scriptsize ${symbol}$}}}}}}')
    for j, (label, value) in enumerate(right):
        y = 173-32*j
        symbol = ['\\bullet','\\square','\\triangle','\\diamond'][j]
        lines.extend([f'\\put(326,{y}){{\\makebox(0,0)[r]{{\\scriptsize {label}}}}}',
                      f'{{\\color{{{colours[j]}}}\\put({330+130*(value+.15)/.35:.2f},{y}){{\\makebox(0,0){{\\scriptsize ${symbol}$}}}}}}'])
    lines.extend([r'\put(205,15){\makebox(0,0){\scriptsize Recovery difference}}',
                  r'\put(390,15){\makebox(0,0){\scriptsize English, Zipf weighted}}', r'\end{picture}',
                  r'\par\small$\bullet$ Original\quad$\square$ 20261007\quad$\triangle$ 20261008\quad$\diamond$ 20261009',
                  r'\caption{Selected partition mean contrasts. (a) Mizo minus English. (b) English length-balanced Zipf query-weighted FLORES minus WMT23. Symbols identify partitions}',
                  r'\label{fig:partition}', r'\end{figure}'])
    return '\n'.join(lines)


def render(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'ps.fonttype':42, 'pdf.fonttype':42})
    left, right = panels(result)
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 6.7))
    fig.subplots_adjust(left=.34, right=.97, bottom=.14, top=.95, hspace=.5)
    colours = ['#333333', '#2166ac', '#b76519', '#23735c']
    for i, (label, values) in enumerate(left):
        for j, (partition, mean) in enumerate(values):
            axes[0].plot(mean, i+(j-1.5)*.12, marker=['o','s','^','D'][j], color=colours[j], label=partition if i==0 else None, markersize=5)
    axes[0].set_yticks(range(len(left)), [x[0] for x in left])
    axes[0].invert_yaxis()
    axes[0].text(0,1.03,'(a)',transform=axes[0].transAxes)
    for j, (_, mean) in enumerate(right):
        axes[1].plot(mean, j, marker=['o','s','^','D'][j], color=colours[j], markersize=6)
    axes[1].set_yticks(range(len(right)), [x[0] for x in right])
    axes[1].invert_yaxis()
    axes[1].text(0,1.03,'(b)',transform=axes[1].transAxes)
    for ax in axes:
        ax.set_xlim(-.15, .2)
        ax.set_xticks([-.1,0,.1,.2])
        ax.axvline(0, color='#aaaaaa', linewidth=.8)
        ax.set_xlabel('Mean recovery difference')
        ax.spines[['top','right']].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.01))
    fig.savefig(ROOT / 'figures/partition-sensitivity.png', dpi=180)
    fig.savefig(ROOT / 'figures/Fig2.eps')
    fig.savefig(ROOT / 'paper/Fig2.pdf')
    plt.close(fig)


def decisions():
    if not (ROOT / 'results/corpus/linguistic-clarification.json').exists():
        return read('results/corpus/linguistic-decisions.json')
    original = read('results/corpus/linguistic-clarification.json')
    match = re.search(r'four of under-stemming \(([^)]+)\) and five of over-stemming \(([^)]+)\)', ' '.join(original['paragraphs']))
    if match is None:
        raise ValueError('Reviewer categories unavailable')
    categories = {token.strip(): category for group, category in zip(match.groups(), ['Under-stemming','Over-stemming']) for token in group.split(',')}
    return {'source_sha256': original['source_sha256'], 'mizo_error_types': categories,
            'scope': 'Accepted error classifications. Private correspondence text was excluded.'}


def main():
    result = build()
    (OUT / 'uncertainty.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    (ROOT / 'results/corpus/linguistic-decisions.json').write_text(json.dumps(decisions(), indent=2, ensure_ascii=False, sort_keys=True)+'\n', encoding='utf-8')
    render(result)
    print('Recomputed seed uncertainty, paired contrasts, structured decisions and partition figure')


if __name__ == '__main__':
    main()
