"""Load the HateXplain dataset.

Source: https://github.com/hate-alert/HateXplain  (Mathew et al., AAAI 2021)

Expected files in data/raw/hatexplain/:
    dataset.json            {post_id: {annotators: [{label,...} x3], post_tokens: [...], ...}}
    post_id_divisions.json  {"train": [ids], "val": [ids], "test": [ids]}

Each post has 3 annotators. The final label is the majority vote. If all three
annotators disagree the post is marked "undecided" (label_binary = NaN) and
should be dropped before training - as in the original paper.

Label mapping (see config.HATEXPLAIN_LABEL_MAP):
    normal -> 0 (Non-Hate) | offensive -> 1 | hatespeech -> 1  (Hate/Abusive)
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from src.data.io_utils import (binarize_label, empty_unified, find_data_files,
                               pick_column, read_any)
from src.utils.config import (HATEXPLAIN_DIR, HATEXPLAIN_LABEL_MAP,
                              LABEL_COL_CANDIDATES, TEXT_COL_CANDIDATES,
                              UNIFIED_COLUMNS)


def _majority(labels: list[str]) -> str:
    counts = Counter(labels).most_common()
    if not counts:
        return "undecided"
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "undecided"  # tie (e.g. 1-1-1)
    return counts[0][0]


def load_hatexplain(root=HATEXPLAIN_DIR, verbose: bool = True) -> pd.DataFrame:
    ds_files = list(root.rglob("dataset.json")) if root.exists() else []
    if ds_files:
        ds_path = ds_files[0]
        with open(ds_path, encoding="utf-8") as f:
            data = json.load(f)

        split_of = {}
        div_files = list(root.rglob("post_id_divisions.json"))
        if div_files:
            with open(div_files[0], encoding="utf-8") as f:
                divisions = json.load(f)
            for split_name, ids in divisions.items():
                key = "val" if split_name in {"val", "valid", "validation"} else split_name
                for pid in ids:
                    split_of[pid] = key
        elif verbose:
            print("[HateXplain] post_id_divisions.json not found -> split='unknown'")

        rows = []
        for pid, rec in data.items():
            labels = [a["label"] for a in rec["annotators"]]
            final = _majority(labels)
            rows.append({
                "text": " ".join(rec["post_tokens"]),
                "label_raw": final,
                "label_binary": HATEXPLAIN_LABEL_MAP.get(final, np.nan),
                "language": "English",
                "source": "HateXplain",
                "split": split_of.get(pid, "unknown"),
            })
        df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
        if verbose:
            print(f"[HateXplain] loaded {ds_path.name}: {len(df):,} posts")
        return df

    # ---- fallback: a flat csv/parquet export (e.g. from Hugging Face) ----
    files = find_data_files(root)
    frames = []
    for f in files:
        df = read_any(f)
        t = pick_column(df, TEXT_COL_CANDIDATES)
        l = pick_column(df, LABEL_COL_CANDIDATES)
        if t is None or l is None:
            continue
        frames.append(pd.DataFrame({
            "text": df[t].astype("string"), "label_raw": df[l],
            "label_binary": binarize_label(df[l]), "language": "English",
            "source": "HateXplain", "split": "unknown"}))
    if not frames:
        print(f"[HateXplain] nothing usable in {root}. Run scripts/download_data.py first.")
        return empty_unified()
    print("[HateXplain] WARNING: flat export used - verify label_raw -> label_binary mapping!")
    return pd.concat(frames, ignore_index=True)[UNIFIED_COLUMNS]
