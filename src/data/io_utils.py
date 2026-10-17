"""Format-agnostic file reading + inspection helpers.

We never assume column names: these helpers read whatever is on disk and let
us look at it first.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.utils.config import (LANGUAGE_ALIASES, UNIFIED_COLUMNS)

DATA_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".parquet"}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}


def find_data_files(root: Path) -> list[Path]:
    """Recursively list readable data files under `root`."""
    root = Path(root)
    if not root.exists():
        return []
    def _visible(p: Path) -> bool:
        rel_parts = p.relative_to(root).parts
        # skip hidden folders (.cache from huggingface_hub, .git, ...) and known junk
        return not any(part.startswith(".") or part in SKIP_DIRS for part in rel_parts)

    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in DATA_SUFFIXES and _visible(p)
    )


def read_any(path: Path) -> pd.DataFrame:
    """Read csv/tsv/json/jsonl/parquet into a DataFrame."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in {".csv", ".tsv"}:
        sep = "\t" if suf == ".tsv" else ","
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(path, sep=sep, encoding=enc)
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                return pd.read_csv(path, sep=sep, encoding=enc,
                                   engine="python", on_bad_lines="skip")
        raise ValueError(f"Could not decode {path}")
    if suf == ".jsonl":
        return pd.read_json(path, lines=True)
    if suf == ".json":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return pd.json_normalize(obj)
        if isinstance(obj, dict):
            vals = list(obj.values())
            if vals and all(isinstance(v, dict) for v in vals):
                return pd.DataFrame.from_dict(obj, orient="index").reset_index(drop=True)
            return pd.json_normalize(obj)
    raise ValueError(f"Unsupported file type: {path}")


def pick_column(df: pd.DataFrame, candidates: Iterable[str],
                override: Optional[str] = None) -> Optional[str]:
    """Return the real column name matching the first candidate (case-insensitive)."""
    if override is not None:
        if override not in df.columns:
            raise KeyError(f"Override column '{override}' not in {list(df.columns)}")
        return override
    lowered = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def detect_language_from_path(path: Path, root: Path) -> Optional[str]:
    """Look for a language name/code in folder names or the file name."""
    try:
        rel = Path(path).relative_to(root)
    except ValueError:
        rel = Path(path)
    tokens = []
    for part in list(rel.parts[:-1]) + [rel.stem]:
        tokens += re.split(r"[^a-zA-Z]+", part.lower())
    for tok in tokens:
        if tok in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[tok]
    return None


def detect_split_from_path(path: Path) -> str:
    stem_tokens = re.split(r"[^a-zA-Z]+", Path(path).stem.lower())
    for tok in stem_tokens:
        if tok == "train":
            return "train"
        if tok in {"val", "valid", "validation", "dev"}:
            return "val"
        if tok == "test":
            return "test"
    return "unknown"


def normalise_language(value) -> str:
    if pd.isna(value):
        return "Unknown"
    s = str(value).strip().lower()
    return LANGUAGE_ALIASES.get(s, str(value).strip().title())


_NEG = ("non", "not", "normal", "neutral", "benign", "clean")
_POS = ("unsafe", "abus", "hate", "offens", "toxic", "harm", "profan")


def binarize_label(series: pd.Series) -> pd.Series:
    """Best-effort map of a raw label column to {0,1} (NaN if unsure).

    ALWAYS verify with the crosstab in notebook 01 - this is a heuristic.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum() and series.notna().any():
        return numeric.where(numeric.isin([0, 1]))

    def _one(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if any(n in s for n in _NEG):
            return 0
        if any(p in s for p in _POS):
            return 1
        if "safe" in s:
            return 0
        return np.nan

    return series.map(_one).astype("float")


def empty_unified() -> pd.DataFrame:
    return pd.DataFrame(columns=UNIFIED_COLUMNS)


def inspect_frame(df: pd.DataFrame, name: str = "", n_head: int = 3) -> None:
    """Print columns, dtypes, shape, nulls, duplicates and a few rows."""
    print(f"=== {name} ===")
    print(f"rows: {len(df):,} | columns: {df.shape[1]}")
    def _nunique(col):
        try:
            return col.nunique(dropna=True)
        except TypeError:  # unhashable cells (lists/dicts)
            return -1

    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "n_unique": [_nunique(df[c]) for c in df.columns],
    })
    print(info.to_string())
    try:
        print("duplicate rows:", int(df.duplicated().sum()))
    except TypeError:  # unhashable cells (lists) in json files
        print("duplicate rows: n/a (unhashable cells)")
    print(df.head(n_head).to_string(max_colwidth=60))
    print()


def read_processed_csv(path) -> pd.DataFrame:
    """Read a CSV written by this project. Empty cells become NaN, nothing else does
    (so a comment that literally says "NA" or "null" is NOT turned into a missing value)."""
    return pd.read_csv(path, encoding="utf-8", keep_default_na=False, na_values=[""])


def write_processed_csv(df: pd.DataFrame, path) -> None:
    """Write UTF-8 CSV (Indic scripts + emojis safe). Do not open it by double-clicking in Excel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
