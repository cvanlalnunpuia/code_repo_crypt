import sys
import re
import unittest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from springer_template import OFFICIAL, preamble, bibliography, journal_settings


class SpringerTemplateTests(unittest.TestCase):
    def test_vendor_files_unchanged(self):
        for name, vendor in [('sn-jnl.cls', 'sn-jnl.cls'), ('sn-mathphys-num.bst', 'bst/sn-mathphys-num.bst')]:
            self.assertEqual((ROOT/'paper'/name).read_bytes(), (OFFICIAL/vendor).read_bytes())

    def test_official_preamble_used(self):
        for name in ['manuscript.tex', 'supplement.tex']:
            self.assertTrue((ROOT/'paper'/name).read_text(encoding='utf-8').startswith(preamble()))

    def test_tilde_in_reference_url(self):
        entry = dict(key='enron', authors='Carnegie Mellon University', title='Enron Email Dataset',
                     venue='Dataset documentation', url='https://www.cs.cmu.edu/~enron/')
        result = bibliography([entry])
        self.assertIn('https://www.cs.cmu.edu/%7Eenron/', result)
        self.assertNotIn('~', result)

    def test_journal_font_settings(self):
        for name in ['manuscript.tex', 'supplement.tex']:
            self.assertIn(journal_settings(), (ROOT/'paper'/name).read_text(encoding='utf-8'))
        self.assertIn(r'\@setfontsize\normalsize{12pt}{14.4pt}', journal_settings())

    def test_confirmed_declarations_and_standalone_article(self):
        text = (ROOT/'paper/manuscript.tex').read_text(encoding='utf-8')
        self.assertIn('No funding was received for this study.', text)
        self.assertIn('The authors declare no competing interests.', text)
        self.assertNotIn('Online Resource', text)

    def test_table_column_counts(self):
        text = (ROOT/'paper/manuscript.tex').read_text(encoding='utf-8')
        for block in re.findall(r'\\begin\{tabularx\}.*?\\end\{tabularx\}', text, re.S):
            header = block.split(r'\toprule', 1)[1].split(r'\midrule', 1)[0]
            self.assertEqual(block.split(r'\toprule', 1)[0].count('arraybackslash'), header.count('&')+1)

    def test_supplement_heading_bookmark(self):
        text = (ROOT/'paper/manuscript.tex').read_text(encoding='utf-8')
        self.assertNotIn(r'\bmhead*{Supplementary information}', text)
        self.assertNotIn(r'\bmhead{Supplementary information}', text)

    def test_portrait_tables_and_conclusion(self):
        text = (ROOT/'paper/manuscript.tex').read_text(encoding='utf-8')
        self.assertNotIn(r'\begin{sidewaystable}', text)
        self.assertEqual(text.count(r'\begin{table}'), 12)
        self.assertIn(r'\section{Conclusion}', text)
        self.assertIn('Statistical significance was not tested in those comparisons.', text)

    def test_percent_in_reference_title(self):
        entry = dict(key='percent', authors='M. Bian', title='Breaking SSE with <1% Leaked Documents',
                     venue='USENIX, 2026', url='https://www.usenix.org/')
        self.assertIn(r'\textless{}1\%', bibliography([entry]))

    def test_portrait_table_values_preserved(self):
        text = (ROOT/'paper/manuscript.tex').read_text(encoding='utf-8')
        ledger = json.loads((ROOT/'paper/numerical-provenance.json').read_text(encoding='utf-8'))
        labels = dict(recovery='recovery', language='language', structure='structure',
                      balanced_selected='balanced', enron_endpoints='ablation', corpora='corpora')
        for key, label in labels.items():
            block = next(block for block in re.findall(r'\\begin\{table\}.*?\\end\{table\}', text, re.S)
                         if r'\label{tab:'+label+'}' in block)
            for row in ledger['table_rows'][key]:
                for value in re.findall(r'-?\d+\.\d+', ' '.join(map(str, row))):
                    self.assertIn(value, block)


if __name__ == '__main__':
    unittest.main()
