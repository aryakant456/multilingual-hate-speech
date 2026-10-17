"""Load the IndicGuard dataset (Hugging Face: l3cube-pune/IndicGuard).

IMPORTANT: IndicGuard is a *LLM safety* corpus (prompts/responses across
Culture-Adaptive, Jailbreaking and Generic Unsafe domains), not a pure
social-media hate corpus. We therefore DO NOT guess which rows are hate.
`label_binary` is derived from the comma-separated `violated_categories` column:
1 if any category is in INDICGUARD_HATE_CATEGORIES (src/utils/config.py), 0 if the
prompt is labelled safe, NaN otherwise (unsafe for non-hate reasons). Verify the
category names in notebook 01, section 9.

Extra columns `category`, `tag` and `label_raw` are preserved for that decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.io_utils import (binarize_label, empty_unified, find_data_files,
                               normalise_language, pick_column, read_any)
from src.utils.config import (CATEGORY_COL_CANDIDATES, INDICGUARD_CATEGORY_COL,
                              INDICGUARD_DIR, INDICGUARD_HATE_CATEGORIES,
                              INDICGUARD_LABEL_COL, INDICGUARD_LANG_COL,
                              INDICGUARD_TEXT_COL, LABEL_COL_CANDIDATES,
                              LANG_COL_CANDIDATES, TEXT_COL_CANDIDATES,
                              UNIFIED_COLUMNS)
from src.data.io_utils import detect_language_from_path, detect_split_from_path


def _has_hate_category(cell, hate_set: set[str]) -> bool:
    """True if ANY comma-separated category in `cell` is in `hate_set`."""
    if cell is None or pd.isna(cell):
        return False
    return any(part.strip().lower() in hate_set for part in str(cell).split(","))


def load_indicguard(root=INDICGUARD_DIR, verbose: bool = True) -> pd.DataFrame:
    files = find_data_files(root)
    if not files:
        print(f"[IndicGuard] No data files in {root}. Run scripts/download_data.py first.")
        return empty_unified().assign(category=pd.Series(dtype="object"), tag=pd.Series(dtype="object"))

    frames = []
    for f in files:
        try:
            df = read_any(f)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[IndicGuard] skip {f.name}: {exc}")
            continue
        t = pick_column(df, TEXT_COL_CANDIDATES, INDICGUARD_TEXT_COL)
        if t is None:
            if verbose:
                print(f"[IndicGuard] skip {f.name}: no text column in {list(df.columns)[:8]}")
            continue
        l = pick_column(df, LABEL_COL_CANDIDATES, INDICGUARD_LABEL_COL)
        c = pick_column(df, CATEGORY_COL_CANDIDATES, INDICGUARD_CATEGORY_COL)
        g = pick_column(df, LANG_COL_CANDIDATES, INDICGUARD_LANG_COL)
        tg = pick_column(df, ["tag"])

        language = (df[g].map(normalise_language) if g is not None
                    else (detect_language_from_path(f, root) or "Unknown"))
        label_raw = df[l] if l is not None else pd.Series([np.nan] * len(df))
        category = df[c].astype("string") if c is not None else pd.Series([pd.NA] * len(df), dtype="string")

        label_binary = pd.Series(np.nan, index=df.index, dtype="float")
        if INDICGUARD_HATE_CATEGORIES and c is not None:
            hate_set = {x.lower() for x in INDICGUARD_HATE_CATEGORIES}
            is_hate = category.map(lambda v: _has_hate_category(v, hate_set)).astype(bool)
            safe_like = (binarize_label(label_raw) == 0).fillna(False).astype(bool)
            label_binary = label_binary.mask(is_hate, 1.0)
            label_binary = label_binary.mask(safe_like & ~is_hate, 0.0)

        out = pd.DataFrame({
            "text": df[t].astype("string"), "label_raw": label_raw,
            "label_binary": label_binary, "language": language,
            "source": "IndicGuard", "split": detect_split_from_path(f),
            "category": category,
            # `tag` = Generic / Jailbreaking / Culture_adaptive (spelling varies per file)
            "tag": (df[tg].astype("string") if tg is not None
                    else pd.Series([pd.NA] * len(df), dtype="string")),
        })
        # (1) the same prompt id appears several times (one row per response variant,
        #     incl. "refusal_data_augmentation") -> keep one row per id.
        id_col = pick_column(df, ["id"])
        if id_col is not None:
            keep = ~df[id_col].duplicated(keep="first")
            if verbose and (~keep).any():
                print(f"[IndicGuard]   {f.name}: dropping {int((~keep).sum()):,} repeated-id rows")
            out = out.loc[keep.values]
        # (2) some prompts are empty / "REDACTED" (self-harm texts withheld by the authors)
        t_clean = out["text"].astype("string").str.strip()
        bad = t_clean.isna() | (t_clean == "") | (t_clean.str.upper() == "REDACTED")
        bad = bad.fillna(True)
        if bad.any():
            if verbose:
                print(f"[IndicGuard]   {f.name}: dropping {int(bad.sum()):,} empty/REDACTED prompts")
            out = out.loc[~bad.values]
        frames.append(out)
        if verbose:
            print(f"[IndicGuard] loaded {f.relative_to(root)} rows={len(out):,} "
                  f"text='{t}' label='{l}' category='{c}' lang='{g}'")

    if not frames:
        return empty_unified().assign(category=pd.Series(dtype="object"), tag=pd.Series(dtype="object"))
    return pd.concat(frames, ignore_index=True)[UNIFIED_COLUMNS + ["category", "tag"]]
