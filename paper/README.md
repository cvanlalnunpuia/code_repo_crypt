# Manuscript review

Open `paper/manuscript.tex` in a LaTeX editor. Open `paper/supplement.tex` for full comparison tables and stemming disagreements. Use `paper/review-checklist.md` to record requested changes. Discover Informatics was selected as the target journal. The supplied six-author order, affiliations and email addresses were included. C Vanlalnunpuia was designated as corresponding author. Author declarations remain pending. The outstanding submission items are listed in `paper/submission-checklist.md`.

The manuscript incorporates C Lalrinawma's clarified execution procedure. Reviewed suffix rules, exceptions, and expected stems were preserved. Current-form exceptions were applied at every pass. Corpus occurrence was excluded from suffix eligibility. The affected index was rerun for execution and independent replay. Unchanged attack inputs were compared exactly with their archived versions before previous results were retained.

The article reports the completed IHOP experiments. The multi-attack panel and auxiliary-knowledge sweep in the original proposal were not completed. Submission requires coauthor review and the target journal's declarations. Author wording and disclosure requirements should be checked against the institution and journal policies.

The review package contains derived results, reviewed processing resources, configurations, source identifiers, and the editable manuscript. Raw corpus text and the private review workbook were excluded. The package supports numerical manuscript reconstruction from recorded results. Attack reruns require the original project, upstream implementation, and separately acquired corpora.

# Reproduction

Run these commands from the package root with Python 3.9.12, NumPy 1.22.3, and SciPy 1.8.0.

```powershell
python scripts/build_manuscript.py
python scripts/validate_manuscript.py
```

The builder regenerates both LaTeX files and the numerical ledger. The validator compares the entire generated content with the recorded source data. `paper/numerical-provenance.json` records source hashes, reported facts, and table rows. `paper/validation.json` records the numerical and mechanical checks.

The LaTeX sources use the official Springer Nature template. Its class and numerical bibliography style were preserved unchanged. Compile with pdfLaTeX and BibTeX. Keep `sn-jnl.cls`, `sn-mathphys-num.bst`, `sn-bibliography.bib`, `Fig1.pdf`, and `Fig2.pdf` beside the manuscript source. Compilation status is recorded in `paper/compilation-status.json`. The current sources require compilation and final PDF inspection before distribution.
