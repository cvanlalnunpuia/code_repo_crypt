from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRIC = 'query_weighted_recovery'


def data():
    directory = ROOT/'results/corpus/expanded-recovery'
    analysis = json.loads((directory/'expanded-analysis.json').read_text(encoding='utf-8'))
    original = json.loads((ROOT/'results/corpus/production-recovery-analysis.json').read_text(encoding='utf-8'))
    lookup = {tuple(row[key] for key in ['family','partition','language','condition','distribution']):row
              for row in analysis['summaries']}
    selected = [(row['condition'], row['distribution'], row['metric']) for row in original['language_comparisons']
                if row['holm_adjusted_p'] < .05]
    original_partition = next(row['partition'] for row in analysis['summaries'] if row['family']=='original_extended')
    original_runs = json.loads((ROOT/'results/corpus/production-fixed-split-experiments.json').read_text(encoding='utf-8'))['runs']
    seeds = analysis['config']['attack_seeds']
    labels = [('Original', original_partition)]+[(str(seed),seed) for seed in analysis['config']['partition_seeds']]
    def original_mean(language, condition, distribution, metric):
        values = [run['scores'][-1][metric] for run in original_runs if run['dataset_id']==language+':'+condition
                  and run['distribution']==distribution and run['seed'] in seeds]
        if len(values) != len(seeds):
            raise ValueError('Original figure seed coverage differs')
        return sum(values)/len(values)
    contrasts = []
    for condition, distribution, metric in selected:
        values = [original_mean('lus',condition,distribution,metric)-original_mean('en',condition,distribution,metric)]
        for _,seed in labels[1:]:
            left = lookup[('partition',seed,'lus',condition,distribution)][metric]['mean']
            right = lookup[('partition',seed,'en',condition,distribution)][metric]['mean']
            values.append(left-right)
        contrasts.append(dict(condition=condition, distribution=distribution, metric=metric, means=values))
    baseline = []
    for corpus in ['flores200','wmt23']:
        runs = json.loads((ROOT/f'results/corpus/{corpus}-length-balanced-experiments.json').read_text(encoding='utf-8'))['runs']
        values = [run['scores'][-1][METRIC] for run in runs if run['dataset_id']=='en:tokenisation_only:k500'
                  and run['distribution']=='zipf_iid' and run['seed'] in seeds]
        if len(values) != len(seeds):
            raise ValueError('Original balanced seed coverage differs')
        baseline.append(sum(values)/len(values))
    balanced = [baseline[0]-baseline[1]]
    for _,seed in labels[1:]:
        balanced.append(lookup[('balanced',seed,'en','flores200','zipf_iid')][METRIC]['mean']-
                        lookup[('balanced',seed,'en','wmt23','zipf_iid')][METRIC]['mean'])
    return dict(partitions=[list(item) for item in labels], attack_seeds=seeds, language_contrasts=contrasts, balanced_means=balanced,
                expanded_analysis_sha256=hashlib.sha256((directory/'expanded-analysis.json').read_bytes()).hexdigest())


def figure_tex():
    return '\n'.join([r'\begin{figure}[!htbp]',r'\centering',
        r'\includegraphics[width=\textwidth]{Fig2.pdf}',
        r'\caption{Expanded partition contrasts. (a) Selected Mizo-minus-English contrasts. (b) English Zipf weighted FLORES-minus-WMT23 contrasts. Symbols identify partitions}',
        r'\label{fig:partition}',r'\end{figure}'])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = data()
    colours = ['#222222','#2166ac','#8c4d10','#23735c','#762a83','#b63c40','#007a87']
    markers = ['o','s','^','D','v','P','X']
    names = {'stemming':'Stemming','stopword_removal':'Stopwords'}
    metric_names = {'query_weighted_recovery':'weighted','distinct_keyword_macro_recovery':'macro'}
    plt.rcParams.update({'font.family':'Arial','font.size':12,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(2,1,figsize=(6.8,7.5))
    fig.subplots_adjust(left=.37,right=.97,bottom=.18,top=.96,hspace=.48)
    for i,row in enumerate(result['language_contrasts']):
        for j,value in enumerate(row['means']):
            axes[0].plot(value,i+(j-3)*.085,marker=markers[j],color=colours[j],markersize=5,
                         linestyle='none',label=result['partitions'][j][0] if i==0 else None)
    labels = [names[row['condition']]+', '+('Zipf' if row['distribution']=='zipf_iid' else 'uniform')+' '+metric_names[row['metric']]
              for row in result['language_contrasts']]
    axes[0].set_yticks(range(len(labels)),labels)
    axes[0].set_ylim(len(labels)-.5,-.5)
    for j,value in enumerate(result['balanced_means']):
        axes[1].plot(value,j,marker=markers[j],color=colours[j],markersize=6,linestyle='none')
    axes[1].set_yticks(range(len(result['partitions'])),[label for label,_ in result['partitions']])
    axes[1].set_ylim(len(result['partitions'])-.5,-.5)
    values = [value for row in result['language_contrasts'] for value in row['means']]+result['balanced_means']
    lower = min(-.15,min(values)-.02)
    upper = max(.2,max(values)+.02)
    for index,ax in enumerate(axes):
        ax.set_xlim(lower,upper)
        ax.set_xticks([-.1,0,.1,.2])
        ax.axvline(0,color='#777777',linewidth=.8)
        ax.set_xlabel('Mean recovery difference')
        ax.spines[['top','right']].set_visible(False)
        ax.text(0,1.03,'('+chr(97+index)+')',transform=ax.transAxes)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.53,.015),fontsize=12)
    for path in ['figures/partition-sensitivity.png','figures/Fig2.eps','paper/Fig2.pdf']:
        fig.savefig(ROOT/path,dpi=180)
    plt.close(fig)
    (ROOT/'results/corpus/expanded-recovery/figure-data.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Rendered expanded partition figure from matched saved attack seeds')


if __name__ == '__main__':
    main()
