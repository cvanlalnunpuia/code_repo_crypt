# Corpus acquisition

The corpus pipeline preserves one record per source document. Each record retains its language, source reference, retrieval time, text hashes, parser version, and rights state. Unicode text was normalised to NFC. Mizo diacritics were retained.

## Sources

The initial source review identified the Mizoram Directorate of Information and Public Relations, Zalen, and Vanglaini as possible Mizo sources. Zalen states that it publishes Mizo and English news on its website and application ([Zalen](https://www.zalen.in/page/about-us)). Vanglaini restricts copying or storage beyond personal use without prior written permission ([Vanglaini terms](https://www.vanglaini.org/terms-of-use)). Vanglaini was excluded until written permission is obtained.

Common Crawl News publishes daily Web ARChive files for news sites from 2016 onward ([Common Crawl](https://commoncrawl.org/blog/news-dataset-available)). The archive remains a candidate English source. The rights attached to each publisher still require review before full text is retained or redistributed.

WMT23 IndicNECorp1.0 was selected for the production English-Mizo comparison. The WMT23 data policy permits research use and requires the applicable citations ([WMT23 data policy](https://www2.statmt.org/wmt23/translation-task.html)). The official English-Mizo archive supplies aligned training, validation, and test files ([WMT23 Indic task](https://www2.statmt.org/wmt23/indic-mt-task.html)). The manifest permits local full-text storage and metadata-only redistribution.

FLORES-200 was selected for external-domain sensitivity analysis. The official repository assigns CC BY-SA 4.0 to the corpus ([FLORES-200 repository](https://github.com/facebookresearch/flores/tree/main/flores200)). The direct Meta archive requires no account or approval. Its English-Mizo files contain material from Wikinews, Wikivoyage, and Wikibooks.

The FLORES-200 acquisition checks the complete archive against a pinned SHA-256 digest. It extracts the English file, Mizo file, and sentence metadata for the dev and devtest splits. Source URLs group aligned sentences into article-level excerpts. The public files contain 2,009 sentence pairs and produce 562 article pairs. The audit is recorded in `results/corpus/flores200-audit.json` and `docs/flores200-corpus-audit.md`.

The archive contained no licence file or README. The external WMT23 data policy and task page provide the recorded permission evidence. The raw text remains local to the project. The archive download replaces website crawling, so a robots review does not apply.

The acquisition script retrieves the six English-Mizo files through byte-range requests. Each ZIP entry is checked against its recorded uncompressed size and CRC-32 value. A SHA-256 digest is recorded after extraction. The archive URL, byte count, ETag, modification time, retrieval time, entry path, and local file digest are stored in `results/corpus/wmt23-acquisition.json`.

The training split contained 50,000 aligned rows. Fourteen rows had empty source text. Fifty-eight rows produced no indexable tokens on at least one side. Another 180 rows repeated an earlier bilingual pair. These rows were excluded before grouping. Ten consecutive valid sentence pairs formed one bilingual document pair. The final incomplete block contained eight rows and was excluded. The production corpus contains 4,974 document pairs and 9,948 language records. Source row numbers are stored for every document.

The first sentence in each block is stored in the title field. The remaining nine sentences are stored in the body field. Both fields enter the index. The grouping preserves every selected sentence once. Validation and test files remain outside the production corpus and are available for preprocessing review.

The acquisition gate checks approval status, permission evidence, full-text storage permission, language, host allowlist, and parser registration. Publisher crawling also requires a robots review. Direct archive downloads use the access terms recorded in the manifest.

DIPR Mizoram remains a candidate source for later external validation. Its adapter and synthetic parser fixtures remain available. Live DIPR collection remains disabled.

The DIPR listing adapter accepts saved category pages. It retains unique `/post/` URLs inside the press-release listing, removes query strings, and sorts the output. Navigation, attachment, and footer links are excluded. The adapter has separate synthetic fixtures for listing discovery and article extraction.

## Records

Raw parser output used JSON Lines. The normaliser removed tracking parameters from canonical URLs, standardised time stamps, normalised whitespace, retained paragraph boundaries, and computed deterministic SHA-256 identifiers. It rejected a source unless the manifest approved both acquisition and full-text storage.

HTML article records use case-folded hashes and trigram Jaccard similarity for duplicate annotation. WMT23 blocks are checked for repeated normalised text before record construction. A repeated block in either language excludes the complete bilingual pair.

## Validation

The fixture pipeline can be run from the project root.

```powershell
.\.venv-py39\Scripts\python.exe scripts\validate_source_manifest.py corpus\sources\source-manifest.json --schema corpus\schema\source-manifest.schema.json
.\.venv-py39\Scripts\python.exe scripts\parse_html_corpus.py corpus\fixtures\article-en.html corpus\fixtures\parsed-article.jsonl --manifest corpus\sources\source-manifest.json --source-id fixture_news --url https://fixtures.invalid/news/parsed --language en --retrieved-at 2026-01-02T00:00:00Z
.\.venv-py39\Scripts\python.exe scripts\normalize_corpus.py corpus\fixtures\raw-articles.jsonl corpus\fixtures\normalised-articles.jsonl --manifest corpus\sources\source-manifest.json
.\.venv-py39\Scripts\python.exe scripts\validate_corpus.py corpus\fixtures\normalised-articles.jsonl --schema corpus\schema\article.schema.json
.\.venv-py39\Scripts\python.exe scripts\deduplicate_corpus.py corpus\fixtures\normalised-articles.jsonl corpus\fixtures\deduplicated-articles.jsonl --threshold 0.75
.\.venv-py39\Scripts\python.exe scripts\validate_corpus.py corpus\fixtures\deduplicated-articles.jsonl --schema corpus\schema\article.schema.json --require-duplicate
.\.venv-py39\Scripts\python.exe -m unittest tests.test_corpus_pipeline
```

The WMT23 production acquisition and validation can be run from the project root.

```powershell
.\.venv-py39\Scripts\python.exe scripts\acquire_wmt23_corpus.py --config configs\wmt23_indicnecorp.json
.\.venv-py39\Scripts\python.exe scripts\build_wmt23_corpus.py --config configs\wmt23_indicnecorp.json
.\.venv-py39\Scripts\python.exe scripts\validate_wmt23_corpus.py --config configs\wmt23_indicnecorp.json
.\.venv-py39\Scripts\python.exe scripts\validate_corpus.py results\corpus\wmt23-records.jsonl --schema corpus\schema\article.schema.json --require-duplicate
```

The FLORES-200 sensitivity corpus can be rebuilt from the project root.

```powershell
.\.venv-py39\Scripts\python.exe scripts\acquire_flores200_corpus.py --config configs\flores200.json
.\.venv-py39\Scripts\python.exe scripts\build_flores200_corpus.py --config configs\flores200.json
.\.venv-py39\Scripts\python.exe scripts\validate_flores200_corpus.py --config configs\flores200.json
.\.venv-py39\Scripts\python.exe scripts\build_corpus_split.py results\corpus\flores200-split.json --config configs\corpus_split_flores200.json
.\.venv-py39\Scripts\python.exe scripts\build_tokenised_corpus.py results\corpus\flores200-records.jsonl results\corpus\flores200-selection.json results\corpus\flores200-tokenisation-only --config configs\tokenisation_only.json
.\.venv-py39\Scripts\python.exe scripts\audit_flores200_corpus.py
```

The stored corpus should exclude page navigation, advertisements, reader comments, and related-article lists. Each source parser requires fixed HTML fixtures and extraction tests before acquisition is enabled. Raw snapshots should be retained only when the manifest permits full-text storage. Derived-only sources may retain hashes, counts, and approved metadata.
