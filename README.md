# Corpus and Preprocessing Effects on IHOP Query Recovery in English and Mizo

This repository provides recorded experiment results, preprocessing resources and reconstruction scripts for the English-Mizo IHOP evaluation. IHOP is a statistical attack that associates encrypted-search observations with candidate plaintext keywords.

## Reproduction

Download `reproducibility-release.zip` and extract it into a new directory. Use Python 3.9.12 and install the packages in `requirements-2022.lock`.

```text
python -m pip install -r requirements-2022.lock
python scripts/setup_official_template.py
python scripts/build_manuscript.py
python scripts/validate_manuscript.py
```

The template setup downloads the original Springer Nature files from the publisher. Their SHA-256 identifiers are checked before use. The builder generates `paper/manuscript.tex`, `paper/supplement.tex` and the numerical provenance ledger. The validator checks reported values and source reconstruction. Final PDF layout requires separate inspection.

Recorded results support numerical reconstruction without raw corpus text. Full attack reruns require separately acquired corpora and the pinned upstream IHOP implementation. Acquisition and experiment instructions are in `docs/corpus-acquisition.md` and `paper/reproduction.md` inside the archive.

The release checks also require the pinned upstream implementation. Synthetic corpus fixtures are included for parser and checkpoint tests. The setup script obtains a source-only sparse checkout and verifies the revision.

```text
python scripts/setup_ihop.py
python scripts/run_release_tests.py
```

The complete integration suite is run with `python -m unittest discover -s tests` after reconstructing the production corpus indexes and acquisition-dependent artifacts. Those inputs are excluded from this archive. The release checks cover the separately specified fixture, recovery-analysis and template families.

## Scope

The original Enron reference discrepancy remained unresolved. Independent checks were performed on small assignment fixtures. Production-scale independent equivalence remained unvalidated. Linguistic agreement was measured on the reviewed development sample.

The public archive excludes raw corpora and private review correspondence. The local review workbook path was removed. Experimental values and linguistic decisions were preserved. Source hashes for the public metadata were regenerated.

## Availability

The repository is available at https://github.com/cvanlalnunpuia/code_repo_crypt. Third-party resources retain their own terms. No additional reuse licence has been assigned to the original project files. See `THIRD_PARTY.md`.
