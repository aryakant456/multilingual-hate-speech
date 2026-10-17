"""Load the MACD (Multilingual Abusive Comment Detection) dataset.

Source: https://github.com/ShareChatAI/MACD  (Gupta et al., NeurIPS 2022)

Facts verified from the real files (see src/utils/config.py):
  * only the folder data/raw/macd/<MACD_SUBDIR>/ (= "dataset") is used; the other
    folders are re-splits of the same comments.
  * columns are `text` and `label`; the language is in the file name
    (hindi_train.csv, malyalam_val.csv [sic], ...).
  * label 0 = ABUSIVE, label 1 = NON-abusive  ->  we convert so that
    label_binary 1 = Hate/Abusive, 0 = Non-Hate. The untouched value stays in label_raw.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.io_utils import (binarize_label, detect_language_from_path,
                               detect_split_from_path, empty_unified,
                               find_data_files, normalise_language, pick_column,
                               read_any)
from src.utils.config import (LANG_COL_CANDIDATES, LABEL_COL_CANDIDATES,
                              MACD_ABUSIVE_LABEL, MACD_DIR, MACD_LABEL_COL,
                              MACD_SUBDIR, MACD_TEXT_COL, TEXT_COL_CANDIDATES,
                              UNIFIED_COLUMNS)


def _macd_binary(raw: pd.Series) -> pd.Series:
    """1 = Hate/Abusive, 0 = Non-Hate, using MACD's own label convention."""
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().all():
        return (num == MACD_ABUSIVE_LABEL).astype(float)
    return binarize_label(raw)  # text labels (not the case for current MACD files)


def load_macd(root=MACD_DIR, verbose: bool = True, subdir: str | None = MACD_SUBDIR) -> pd.DataFrame:
    root = Path(root)
    base = root / subdir if subdir and (root / subdir).exists() else root
    files = find_data_files(base)
    if not files:
        print(f"[MACD] No data files found in {base}. Run scripts/download_data.py first.")
        return empty_unified()
    if verbose:
        print(f"[MACD] reading {len(files)} file(s) from {base.relative_to(root.parent)}  "
              f"(label {MACD_ABUSIVE_LABEL} = abusive -> label_binary 1)")

    frames = []
    for f in files:
        try:
            df = read_any(f)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[MACD] skip {f.name}: cannot read ({exc})")
            continue

        text_col = pick_column(df, TEXT_COL_CANDIDATES, MACD_TEXT_COL)
        label_col = pick_column(df, LABEL_COL_CANDIDATES, MACD_LABEL_COL)
        if text_col is None or label_col is None:
            if verbose:
                print(f"[MACD] skip {f.relative_to(root)}: no text/label column "
                      f"(columns={list(df.columns)[:8]})")
            continue

        lang_col = pick_column(df, LANG_COL_CANDIDATES)
        if lang_col is not None:
            language = df[lang_col].map(normalise_language)
        else:
            language = detect_language_from_path(f, root) or "Unknown"

        out = pd.DataFrame({
            "text": df[text_col].astype("string"),
            "label_raw": df[label_col],
            "label_binary": _macd_binary(df[label_col]),
            "language": language,
            "source": "MACD",
            "split": detect_split_from_path(f),
        })
        frames.append(out[UNIFIED_COLUMNS])
        if verbose:
            print(f"[MACD] loaded {f.relative_to(root)}  rows={len(out):,} "
                  f"text='{text_col}' label='{label_col}' language={out['language'].iloc[0]}")

    if not frames:
        return empty_unified()
    return pd.concat(frames, ignore_index=True)
