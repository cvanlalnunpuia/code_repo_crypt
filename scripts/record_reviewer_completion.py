from __future__ import annotations

import hashlib
import json
from pathlib import Path
import argparse
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-exit-code', type=int)
    parser.add_argument('--compiler-diagnostics', default='Unable to find standard directories for platform')
    args = parser.parse_args()
    if args.test_exit_code is not None and args.test_exit_code != 0:
        raise ValueError('Test run failed')
    ledger = json.loads((ROOT / 'paper/numerical-provenance.json').read_text(encoding='utf-8'))
    facts = ledger['facts']
    value = lambda name: facts[name]['rendered']
    record = [
        '# Revision', '',
        f"The revision included {value('SensitivityRuns')} additional descriptive recovery runs. Original production results, comparison families, and linguistic decisions were preserved. Source reconstruction and all tests passed. The figures were visually inspected. Both standalone sources were submitted to the built-in compiler. Compilation remained unverified because its runtime was unavailable.", '',
        '1. The Enron reference mismatch remained unresolved. An independent binomial-likelihood calculation and exhaustive assignment check passed on the planted fixture. Execution replay established repeatability within the pinned implementation. The manuscript retained the distinction.', '',
        f"2. Three additional paired partitions were evaluated with three attack seeds per condition. The original positive Mizo-minus-English direction was retained in {value('SensitivityPositiveDirections')} of {value('SensitivityDirectionCount')} additional means. English length-balanced Zipf contrasts ranged from {value('BalancedSensitivityMin')} to {value('BalancedSensitivityMax')}. The abstract and Discussion were revised to state partition dependence.", '',
        f"3. A controlled set of {value('ControlledKeywords')} word labels per language was evaluated. Output stems were unique among selected labels. The restricted tokenisation-only and stopword-only matrices were identical. Controlled Zipf stemming-minus-stopword differences were {value('ENControlledStemZipfDifference')} in English and {value('LUSControlledStemZipfDifference')} in Mizo. Vocabulary overlap and stem-merger counts were added to the supplement.", '',
        f"4. Query coverage was reported. Zipf runs observed {value('ZipfObservedMin')}-{value('ZipfObservedMax')} keywords, with a mean of {value('ZipfObservedMean')}. Shuffled-rank controls were compared with frequency-ranked queries on matched attack seeds. The accompanying results retained the separate weighted and macro measurements.", '',
        '5. The complete reviewed suffix inventory, execution priorities, minimum lengths, replacements, exceptions, and excluded stopwords were added. The coauthor development-review role and purposive annotation selection were described. Existing labels were preserved. Agreement was reported on the reviewed development sample. Additional linguistic review was excluded from the submission requirements.', '',
        'Confidence intervals were labelled unadjusted. IHOP terminology was corrected. Corpus-control preprocessing, matching objectives, partition seeds, and reconstruction requirements were specified. Figure labels and transformed iteration spacing were corrected. The supplied author order and contact details were included.', '',
        f"The full commit identifier was given line-break opportunities. The bibliography contained {len(ledger['references'])} cited sources in citation order. The data availability statement distinguished numerical reconstruction from full attack reruns. The official Springer Nature template was used with its unchanged class and numerical bibliography style. Methods preceded Discussion, following the sample template. Long reference URLs were given line-break opportunities.", '',
        'The Enron reference URL used percent encoding for its tilde. This prevented the supplied bibliography style from inserting a text command into the URL. Draft-mode pdfLaTeX checks completed for the article with its regenerated bibliography and for the supplementary material. Script-font size substitutions remained warnings. Final PDF layout verification remained pending.', '',
        'The complete generated manuscript and supplementary material were inspected. The revised Overleaf and numerical reconstruction archives were verified against their source files. Source text and private review correspondence were excluded. Accepted linguistic decisions were retained in structured form.', '',
        'A further review moved partition contrasts beside the original comparisons. A partition figure and within-partition seed standard deviations were added. Paired contrast values were retained with the derived results. Snowball implementation details and dataset citations were added.', '',
        f"Independent likelihood and exhaustive assignment checks passed on {value('RealCheckCases')} real-data subsets. Every recorded update was audited in {value('IterativeCases')} fixture runs through {value('IterativeSteps')} iterations. Production-scale independent equivalence and the Enron reference mismatch remained unresolved. The fixed single-permutation rank control retained its stated scope.", '',
        'Numerical statements in this record were generated from paper/numerical-provenance.json by scripts/record_reviewer_completion.py.', '',
    ]
    (ROOT / 'docs/reviewer-revision.md').write_text('\n'.join(record), encoding='utf-8')
    compile_record = {'state':'unverified', 'reason':'Compiler runtime unavailable',
                      'diagnostics':args.compiler_diagnostics,
                      'paths':['paper/manuscript.tex','paper/supplement.tex'],
                      'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['paper/manuscript.tex','paper/supplement.tex']}}
    (ROOT / 'paper/compilation-status.json').write_text(json.dumps(compile_record, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    path = ROOT / 'paper/release-verification.json'
    release = json.loads(path.read_text(encoding='utf-8'))
    release['source_count'] = len(ledger['sources'])
    release['reported_fact_count'] = len(ledger['facts'])
    release['artifacts'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['paper/manuscript.tex','paper/supplement.tex','figures/production-recovery.png','figures/partition-sensitivity.png','figures/Fig1.eps','figures/Fig2.eps','paper/validation.json','paper/sn-jnl.cls','paper/sn-mathphys-num.bst','paper/sn-bibliography.bib','paper/Fig1.pdf','paper/Fig2.pdf']}
    if args.test_exit_code is not None:
        release['test_count'] = unittest.defaultTestLoader.discover(str(ROOT/'tests')).countTestCases()
        release['test_exit_code'] = args.test_exit_code
    release['native_compilation'] = 'unverified, compiler runtime unavailable'
    release['unresolved'] = ['Enron upstream reference mismatch', 'Production-scale independent attack equivalence', 'Author declarations and approval', 'Compilation and final PDF layout']
    release['figure_visually_inspected'] = True
    release['complete_sources_inspected'] = True
    path.write_text(json.dumps(release, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    status_path = ROOT / 'results/corpus/reviewer-sensitivity/revision-status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    status.update(stage='complete', state='verified_with_compilation_limitation', compilation='unverified', figure_visually_inspected=True, complete_sources_inspected=True, test_count=release['test_count'])
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
