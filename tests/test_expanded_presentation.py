import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from expanded_partition_figure import data, figure_tex
from standalone_manuscript import source_rows, paired, RELEASE_URL


class ExpandedPresentationTests(unittest.TestCase):
    def test_figure_data_reconstructs(self):
        saved = json.loads((ROOT/'results/corpus/expanded-recovery/figure-data.json').read_text(encoding='utf-8'))
        self.assertEqual(data(), saved)

    def test_figure_seed_and_partition_coverage(self):
        saved = data()
        config = json.loads((ROOT/'configs/expanded_recovery.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['attack_seeds'], config['attack_seeds'])
        self.assertEqual([seed for _,seed in saved['partitions'][1:]], config['partition_seeds'])
        self.assertTrue(all(len(row['means'])==len(saved['partitions']) for row in saved['language_contrasts']))

    def test_expanded_control_values_and_prose(self):
        ledger = json.loads((ROOT/'paper/numerical-provenance.json').read_text(encoding='utf-8'))
        analysis = json.loads((ROOT/'results/corpus/expanded-recovery/expanded-analysis.json').read_text(encoding='utf-8'))
        for lang,prefix in [('en','EN'),('lus','LUS')]:
            rows = {row['condition']:row for row in analysis['summaries'] if row['family']=='controlled'
                    and row['language']==lang and row['distribution']=='zipf_iid'}
            difference = rows['stemming']['query_weighted_recovery']['mean']-rows['stopword_removal']['query_weighted_recovery']['mean']
            self.assertEqual(ledger['facts']['Expanded'+prefix+'ControlledDifference']['value'], difference)
        text = (ROOT/'paper/manuscript.tex').read_text(encoding='utf-8')
        self.assertIn('Both differences were close to zero.', text)
        self.assertIn('reproducibility-release.zip', text)
        self.assertIn(RELEASE_URL, text)
        self.assertNotIn('Public deposition of the expanded reconstruction records remained pending.', text)
        self.assertNotIn('Supplementary Table', text)
        self.assertNotIn('Publication of that expanded archive remained pending.', text)

    def test_figure_caption(self):
        self.assertIn('Expanded partition contrasts', figure_tex())
        self.assertIn(r'\label{fig:partition}', figure_tex())
        self.assertIn(r'\begin{figure}[!htbp]', figure_tex())

    def test_standalone_controls_preserve_all_measurements(self):
        article = (ROOT/'paper/manuscript.tex').read_text(encoding='utf-8')
        supplement = (ROOT/'paper/supplement.tex').read_text(encoding='utf-8')
        for number, labels in [(18, ['auxiliary', 'source-control']), (21, ['candidate-control'])]:
            import re
            blocks = [block for block in re.findall(r'\\begin\{table\}.*?\\end\{table\}', article, re.S)
                      if any(r'\label{tab:'+label+'}' in block for label in labels)]
            for row in source_rows(supplement, number):
                self.assertIn(' & '.join(row[-3:]) + r' \\', '\n'.join(blocks))

    def test_standalone_original_contrasts_preserved(self):
        article = (ROOT/'paper/manuscript.tex').read_text(encoding='utf-8')
        supplement = (ROOT/'paper/supplement.tex').read_text(encoding='utf-8')
        rows = source_rows(supplement, 1)
        self.assertEqual(len(rows), 24)
        for first, second in zip(rows[::2], rows[1::2]):
            for column in [5, 6, 7]:
                self.assertIn(paired(first[column], second[column]), article)


if __name__ == '__main__':
    unittest.main()
