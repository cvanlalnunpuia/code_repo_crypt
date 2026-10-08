import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from springer_template import OFFICIAL, preamble, bibliography


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


if __name__ == '__main__':
    unittest.main()
