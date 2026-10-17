"""PHASE 4 - Text preprocessing (two deliberately different pipelines).

Run from the project root (after Phase 3):
    python -m src.data.preprocess

PIPELINE A - XLM-R  (function `clean_for_xlmr`)  ->  column `text`
    MINIMAL cleaning only, because a transformer needs the original wording:
      * decode HTML entities (&amp; -> &), delete real HTML tags (<br>, <b> ...)
      * URLs -> "http", @mentions -> "@user"   (MACD already uses "@user")
      * delete corrupted / invisible characters (U+FFFD, control chars, zero-width space, BOM ...)
      * Unicode NFC normalisation, collapse whitespace
    KEPT ON PURPOSE: upper/lower case, emojis, #hashtags, punctuation ("!!!", "?"),
    every word (no stop-word removal), no stemming, no lemmatisation, and the
    zero-width joiners U+200C / U+200D that are PART of Malayalam / Hindi / Tamil spelling.

PIPELINE B - TF-IDF  (function `clean_for_tfidf`)  ->  column `text_tfidf`
    Starts from pipeline A, then the usual "classical" normalisation:
      lower-case, squeeze letters repeated 3+ times to 2 ("sooooo" -> "soo"),
      punctuation -> space, emojis kept as separate tokens.
    Python's default \\w tokenizer cuts Indic words at their vowel signs, so the TF-IDF
    model uses `whitespace_tokenize` instead.

Then (both pipelines): drop empty rows, drop rows without any letter/digit/emoji,
drop contradicting duplicates, keep one copy of exact duplicates.
"""
from __future__ import annotations

import html
import json
import re
import unicodedata

import pandas as pd

from src.data.io_utils import read_processed_csv, write_processed_csv
from src.utils.config import (CLEAN_FILE, INDICGUARD_CLEAN_FILE,
                              INDICGUARD_UNIFIED_FILE, METRICS_DIR, SPLIT_RANK,
                              UNIFIED_FILE, ZERO_SHOT_LANGUAGES)

# ----------------------------------------------------------------- patterns
_HTML_TAG_RE = re.compile(
    r"</?(?:a|abbr|b|big|blockquote|body|br|center|code|div|em|font|h[1-6]|head|hr|html|i|img|"
    r"li|link|meta|ol|p|pre|script|small|span|strike|strong|style|sub|sup|table|tbody|td|th|tr|"
    r"tt|u|ul)(?:\s[^<>]*)?/?>", re.I)
_USER_PLACEHOLDER_RE = re.compile(r"<user>", re.I)     # HateXplain placeholder
_URL_PLACEHOLDER_RE = re.compile(r"<url>", re.I)       # HateXplain placeholder
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
_MENTION_RE = re.compile(r"@\S+")
# invisible / corrupted characters. NOT removed: U+200C (ZWNJ) and U+200D (ZWJ) - needed by Indic scripts.
_BAD_CHARS_RE = re.compile(
    "[\u200b\u2060\ufeff\u00ad\ufffd\u200e\u200f\u202a-\u202e\u2066-\u2069"
    "\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WS_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"(.)\1{2,}", re.S)


# --------------------------------------------------------------- pipeline A
def clean_for_xlmr(text) -> str:
    """Minimal cleaning for the XLM-R tokenizer. Returns '' for missing input."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    t = str(text)
    t = html.unescape(t)
    t = _USER_PLACEHOLDER_RE.sub("@user", t)
    t = _URL_PLACEHOLDER_RE.sub("http", t)
    t = _HTML_TAG_RE.sub(" ", t)
    t = _URL_RE.sub("http", t)
    t = _MENTION_RE.sub("@user", t)
    t = _BAD_CHARS_RE.sub("", t)
    t = unicodedata.normalize("NFC", t)
    return _WS_RE.sub(" ", t).strip()


# --------------------------------------------------------------- pipeline B
def clean_for_tfidf(text, already_clean: bool = False) -> str:
    """Classical normalisation for TF-IDF. Output is space-separated tokens."""
    t = text if already_clean else clean_for_xlmr(text)
    t = _REPEAT_RE.sub(r"\1\1", t.lower())
    out = []
    for ch in t:
        cat = unicodedata.category(ch)
        if ch in "\ufe0e\ufe0f":                       # emoji variation selectors
            continue
        if cat[0] == "P" or cat in ("Sm", "Sc", "Sk", "Zs", "Zl", "Zp", "Cc"):
            out.append(" ")                            # punctuation / symbols / spaces
        elif cat == "So":
            out.append(f" {ch} ")                      # emoji = its own token
        else:
            out.append(ch)                             # letters, digits, Indic vowel signs ...
    tokens = [tok.strip("\u200c\u200d") for tok in "".join(out).split()]
    return " ".join(tok for tok in tokens if tok)


def whitespace_tokenize(text: str) -> list[str]:
    """Tokenizer for TfidfVectorizer (text_tfidf is already space separated)."""
    return text.split()


def _has_content(text: str) -> bool:
    """True if the text has at least one letter, digit or emoji."""
    return any(ch.isalnum() or unicodedata.category(ch) == "So" for ch in text)


# ------------------------------------------------------------ whole table
def preprocess_frame(df: pd.DataFrame, name: str = "") -> tuple[pd.DataFrame, dict]:
    """Clean a unified table. Returns (clean table, report)."""
    rep = {"table": name, "rows_in": int(len(df))}
    d = df.copy()
    d["text_raw"] = d["text"].astype(str)
    d["text"] = d["text_raw"].map(clean_for_xlmr)

    empty = d["text"] == ""
    rep["dropped_empty_after_cleaning"] = int(empty.sum())
    d = d.loc[~empty]

    no_content = ~d["text"].map(_has_content)
    rep["dropped_no_letter_digit_or_emoji"] = int(no_content.sum())
    d = d.loc[~no_content]

    # ---- duplicates (case-insensitive, across ALL languages / splits) ----
    d["_key"] = d["text"].str.casefold()
    n_labels = d.groupby("_key")["label"].transform("nunique")
    conflicting = n_labels > 1
    rep["dropped_duplicates_with_conflicting_labels"] = int(conflicting.sum())
    d = d.loc[~conflicting]

    # keep the copy that is most useful: training languages before zero-shot languages,
    # then train < val < test.  -> a text can never end up in train AND in test.
    d["_prio"] = (d["language"].isin(ZERO_SHOT_LANGUAGES).astype(int) * 10
                  + d["split"].map(SPLIT_RANK).fillna(5))
    d = d.sort_values("_prio", kind="stable")
    dup = d.duplicated("_key", keep="first")
    rep["dropped_exact_duplicates"] = int(dup.sum())
    d = d.loc[~dup].sort_index()

    d["text_tfidf"] = d["text"].map(lambda s: clean_for_tfidf(s, already_clean=True))
    d = d.drop(columns=["_key", "_prio"]).reset_index(drop=True)
    rep["rows_out"] = int(len(d))
    return d, rep


def main(verbose: bool = True) -> dict:
    if not UNIFIED_FILE.exists():
        raise SystemExit("Run Phase 3 first:  python -m src.data.combine_datasets")
    reports = {}

    clean, reports["train_pool"] = preprocess_frame(read_processed_csv(UNIFIED_FILE), "MACD+HateXplain")
    cols = ["text", "text_tfidf", "label", "language", "source", "split", "label_raw", "text_raw"]
    write_processed_csv(clean[cols], CLEAN_FILE)

    ext, reports["indicguard"] = preprocess_frame(read_processed_csv(INDICGUARD_UNIFIED_FILE), "IndicGuard")
    write_processed_csv(ext[cols + ["category", "tag"]], INDICGUARD_CLEAN_FILE)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    if verbose:
        for r in reports.values():
            print(f"\n[{r['table']}]")
            for k, v in r.items():
                if k != "table":
                    print(f"  {k:<48}{v:>10,}")
        print(f"\nWritten: {CLEAN_FILE}\n         {INDICGUARD_CLEAN_FILE}")
    return reports


if __name__ == "__main__":
    main()
