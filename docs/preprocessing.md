# Preprocessing (Phase 4): two separate pipelines

Code: `src/data/preprocess.py`. Run: `python -m src.data.preprocess`.

## Pipeline A - XLM-R (column `text`): MINIMAL
Why: a transformer reads the whole sentence; capital letters, emojis and punctuation are part of the meaning.
`"I HATE this!!!"` must stay `"I HATE this!!!"`.

Done: decode HTML entities, remove real HTML tags, URLs -> `http`, @mentions -> `@user`, delete corrupted/invisible
characters (U+FFFD, control characters, zero-width space, BOM), Unicode NFC, collapse whitespace.

NOT done (on purpose): stemming, lemmatisation, stop-word removal, lower-casing, removing punctuation/emojis/hashtags.
The zero-width joiners U+200C / U+200D are kept - they are part of Malayalam/Hindi spelling.
Then the text goes straight into the XLM-R tokenizer.

## Pipeline B - TF-IDF (column `text_tfidf`): classical
Starts from pipeline A, then: lower-case, letters repeated 3+ times squeezed to 2 (`sooooo` -> `soo`), punctuation -> space,
emojis kept as separate tokens. Tokenised on whitespace (`whitespace_tokenize`) because Python's default word pattern splits
Hindi/Tamil/Malayalam words at their vowel signs.

## Rows removed (both pipelines)
empty after cleaning; no letter/digit/emoji at all; the same text with contradicting labels (all copies);
exact duplicates (one copy kept: training languages first, then train < val < test, so a text can never be in train and test).
Counts: `results/metrics/preprocessing_report.json`.
