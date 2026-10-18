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

The original Enron numerical reference remained unmatched. Independent binomial calculations and exhaustive assignment minima were checked on small binary fixtures. Every original production update was subsequently audited using independent likelihood costs and assignment-objective calculations. Recovery scores matched the original checkpoints. The SciPy assignment library was shared. Separate independent trajectories were reported without requiring identical tied assignments. Stemming agreement used the completed reviewed development sample.

Expanded results were recorded under `results/corpus/expanded-recovery`. The frozen manifest contained every task and input hash. The configuration was `configs/expanded_recovery.json`. The original confirmatory families were unchanged. The twenty-seed extension used separate exploratory families. `scripts/build_manuscript.py` recomputed the expanded means and standard deviations from checkpoint records. It also recomputed the exploratory probabilities and checked production audit summaries against the original scores. Singleton sample standard deviations were represented as null. `summary-format-provenance.json` recorded that formatting correction.

Full expanded attack reruns required reconstructed raw-corpus exports. The recorded manifest contained acquisition-dependent local export paths. Those paths must be recreated or the manifest regenerated from the same configuration in a separately archived rerun directory. The recorded checkpoint and manifest must remain together. Numerical manuscript reconstruction required only the recorded results supplied in this archive.

The main partition figure used matched attack seeds for the original partition and all expanded partitions. Run `python scripts/expanded_partition_figure.py` to reconstruct its data and vector figures. Its plotted values were recorded in `results/corpus/expanded-recovery/figure-data.json`. Earlier three-seed contrasts were reconstructed by `scripts/review_uncertainty.py` and retained in the supplementary tables. Running that earlier script also regenerates the historical figure. Run the expanded figure script afterward to restore the current Figure 2. The small-fixture iterative audit required the pinned upstream implementation and was run with `scripts/check_iterative_assignment.py`. Both audit outputs were included in the source ledger.
