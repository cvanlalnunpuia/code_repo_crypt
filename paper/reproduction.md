# Reproduction

The accompanying archive contains derived results and source identifiers. Numerical manuscript reconstruction requires Python 3.9.12, NumPy 1.22.3, and SciPy 1.8.0. Run the following commands from the extracted archive root.

```powershell
python scripts/build_manuscript.py
python scripts/validate_manuscript.py
```

The builder regenerates the article, supplementary material, and numerical provenance ledger. The validator compares the entire output with the regenerated content. Derived result files support this reconstruction without raw corpus text.

The original Springer Nature template was included for source reconstruction. The supplied class and numbered bibliography style were preserved unchanged. Both documents require pdfLaTeX. The article also requires BibTeX. The class, bibliography style, bibliography data, and external figure PDFs must remain beside the source files.

Attack reruns require the separately acquired WMT23 IndicNE-Corp and FLORES-200 corpora. Source locations and file names were recorded in `configs/wmt23_indicnecorp.json` and `configs/flores200.json`. Acquisition and construction commands were supplied in `docs/corpus-acquisition.md`. The upstream IHOP repository and revision were recorded in the manuscript. The canonical dependencies were listed in `requirements-2022.lock`.

Production preprocessing and attack exports were configured in `configs/ihop_corpus_export_production.json`. The source pipeline used `build_corpus_split.py`, `build_tokenised_corpus.py`, `build_stopword_corpus.py`, `build_stemmed_corpus.py`, and `export_ihop_corpus.py`. Each script exposed its input paths and configuration through `--help`. Reviewed rules and stopword resources were included. The source corpus files were excluded from redistribution.

Post-review controls were configured in `configs/reviewer_sensitivity.json`. Reconstruct the input exports and run the checkpointed experiment with the following commands after corpus construction.

```powershell
python scripts/reviewer_sensitivity.py --prepare
python scripts/reviewer_sensitivity.py
python scripts/finalize_reviewer_revision.py
```

Existing checkpoint files require the matching manifest and attack exports. Remove no checkpoint until its experiment has been archived. The preparation command regenerates the manifest. Run it only when the intended inputs have been reconstructed. The original confirmatory results and their comparison families remain separate from the post-review descriptive controls.

The original Enron numerical reference remained unmatched. Independent binomial calculations and exhaustive assignment minima were checked on small binary fixtures. Every recorded iterative update was audited. An independently selected trajectory and production-scale equivalence remained unvalidated. Stemming agreement used the completed reviewed development sample.

The partition figure used matched attack seeds for the original and additional partitions. Its underlying paired values and standard deviations were reconstructed by `scripts/review_uncertainty.py`. The small-fixture iterative audit required the pinned upstream implementation and was run with `scripts/check_iterative_assignment.py`. Both audit outputs were included in the source ledger.
