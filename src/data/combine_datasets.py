"""PHASE 3 - Label normalisation: one unified schema for every source.

Run from the project root:
    python -m src.data.combine_datasets

Unified schema (what every later phase uses)
    text      the comment / post
    label     0 = NON-HATE / NON-ABUSIVE      1 = HATE / ABUSIVE
    language  English | Hindi | Tamil | Malayalam | Telugu | Kannada | ...
    source    MACD | HateXplain | IndicGuard
    split     official split of the source dataset (train / val / test)
    label_raw the ORIGINAL label, kept for traceability

How each source label is converted (full explanation: docs/label_mapping.md)
    MACD        raw 0 -> 1 (abusive)      raw 1 -> 0 (non-abusive)     [inverted on purpose]
    HateXplain  hatespeech -> 1, offensive -> 1, normal -> 0, undecided -> dropped
    IndicGuard  "Hate/Identity Hate" in violated_categories -> 1,
                prompt labelled safe -> 0, unsafe for other reasons -> dropped

Nothing is hard-coded blindly: `verify_label_encodings` looks at the labels that are
really in the downloaded files and STOPS with an error if anything unexpected shows up.
"""
from __future__ import annotations

import json

import pandas as pd

from src.data.io_utils import write_processed_csv
from src.data.load_hatexplain import load_hatexplain
from src.data.load_indicguard import load_indicguard
from src.data.load_macd import load_macd
from src.utils.config import (HATEXPLAIN_LABEL_MAP, INDICGUARD_HATE_CATEGORIES,
                              INDICGUARD_UNIFIED_FILE, LABEL_NAMES,
                              MACD_ABUSIVE_LABEL, MACD_PAPER_COUNTS,
                              METRICS_DIR, UNIFIED_FILE)

CORE_COLUMNS = ["text", "label", "language", "source", "split", "label_raw"]


# --------------------------------------------------------------------------- #
# 1. verify the label encodings that are REALLY in the downloaded files
# --------------------------------------------------------------------------- #
def _crosstab_dict(df: pd.DataFrame) -> dict:
    """{raw label: {"1": n, "0": n, "dropped": n}} - shows exactly what each raw value became."""
    lab = df["label_binary"].map({1.0: "1", 0.0: "0"}).fillna("dropped")
    ct = pd.crosstab(df["label_raw"].astype(str), lab)
    for col in ["0", "1", "dropped"]:
        if col not in ct.columns:
            ct[col] = 0
    return {str(k): {c: int(v[c]) for c in ["0", "1", "dropped"]} for k, v in ct.iterrows()}


def verify_label_encodings(macd: pd.DataFrame, hx: pd.DataFrame, ig: pd.DataFrame,
                           verbose: bool = True) -> dict:
    """Check the raw labels of each source. Raises ValueError on anything unexpected."""
    report: dict = {}

    # ---- MACD ----------------------------------------------------------------
    raw = {int(x) for x in pd.to_numeric(macd["label_raw"], errors="coerce").dropna().unique()}
    if not raw <= {0, 1}:
        raise ValueError(f"MACD: expected raw labels {{0, 1}} but found {sorted(raw)}. "
                         "Re-inspect the data before continuing.")
    n_abusive = int((macd["label_raw"].astype(int) == MACD_ABUSIVE_LABEL).sum())
    n_total = len(macd)
    matches_paper = (n_abusive == MACD_PAPER_COUNTS["abusive"]
                     and n_total - n_abusive == MACD_PAPER_COUNTS["non_abusive"])
    report["MACD"] = {
        "raw_values_found": sorted(raw),  # plain python ints
        "rule": f"raw {MACD_ABUSIVE_LABEL} = ABUSIVE -> label 1 ; the other value -> label 0",
        "abusive_rows": n_abusive, "non_abusive_rows": n_total - n_abusive,
        "matches_paper_counts_74550_77872": matches_paper,
    }

    # ---- HateXplain ---------------------------------------------------------
    raw_hx = set(hx["label_raw"].astype(str).unique())
    allowed = set(HATEXPLAIN_LABEL_MAP) | {"undecided"}
    if not raw_hx <= allowed:
        raise ValueError(f"HateXplain: unexpected raw labels {sorted(raw_hx - allowed)}")
    report["HateXplain"] = {
        "raw_values_found": sorted(raw_hx),
        "rule": "hatespeech -> 1, offensive -> 1, normal -> 0, undecided (3 annotators all differ) -> dropped",
    }

    # ---- IndicGuard -------------------------------------------------------------
    raw_ig = set(ig["label_raw"].dropna().astype(str).unique())
    cats = set(ig["category"].astype("string").str.split(",").explode().str.strip().str.lower().dropna().unique())
    missing = {c for c in INDICGUARD_HATE_CATEGORIES if c.lower() not in cats}
    if missing:
        raise ValueError(f"IndicGuard: hate categories {sorted(missing)} do not occur in the data - "
                         "label mapping would produce no hate rows. Fix INDICGUARD_HATE_CATEGORIES.")
    report["IndicGuard"] = {
        "raw_values_found": sorted(raw_ig),
        "rule": f"any of {sorted(INDICGUARD_HATE_CATEGORIES)} in violated_categories -> 1; "
                "prompt_label 'safe' (no hate category) -> 0; other unsafe -> dropped",
    }

    if verbose:
        print("=" * 78)
        print("LABEL ENCODINGS FOUND IN THE REAL FILES")
        for name, r in report.items():
            print(f"\n{name}\n  raw values : {r['raw_values_found']}\n  rule       : {r['rule']}")
        m = report["MACD"]
        print(f"\n  MACD check : {m['abusive_rows']:,} abusive / {m['non_abusive_rows']:,} non-abusive rows "
              f"-> {'MATCHES' if m['matches_paper_counts_74550_77872'] else 'DIFFERS FROM'} the paper "
              f"({MACD_PAPER_COUNTS['abusive']:,} / {MACD_PAPER_COUNTS['non_abusive']:,})")
        if not m["matches_paper_counts_74550_77872"]:
            print("  !! If you did not delete/modify MACD files, re-check the label direction.")
        print("=" * 78)
    return report


# --------------------------------------------------------------------------- #
# 2. build the unified tables
# --------------------------------------------------------------------------- #
def _to_unified(df: pd.DataFrame, extra: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Rename label_binary -> label, drop rows without a usable label/text, fix dtypes."""
    n0 = len(df)
    no_label = int(df["label_binary"].isna().sum())
    text_ok = df["text"].astype("string").str.strip().fillna("") != ""
    no_text = int((~text_ok).sum())
    keep = df["label_binary"].notna() & text_ok
    out = df.loc[keep].copy()
    out["label"] = out["label_binary"].astype(int)
    out["label_raw"] = out["label_raw"].astype(str)
    out["text"] = out["text"].astype(str)
    cols = CORE_COLUMNS + (extra or [])
    return out[cols].reset_index(drop=True), {"rows_in": n0, "dropped_no_label": no_label,
                                              "dropped_no_text": no_text, "rows_out": len(out)}


def build_unified(verbose: bool = True) -> dict:
    macd = load_macd(verbose=False)
    hx = load_hatexplain(verbose=False)
    ig = load_indicguard(verbose=False)
    if macd.empty or hx.empty:
        raise SystemExit("MACD or HateXplain missing - run scripts/download_data.py first (Phases 1-2).")
    if ig.empty:
        raise SystemExit("IndicGuard missing - run: python scripts/download_data.py indicguard")

    verification = verify_label_encodings(macd, hx, ig, verbose=verbose)

    # label mapping evidence (crosstab raw -> unified) BEFORE rows are dropped
    mapping_counts = {"MACD": _crosstab_dict(macd), "HateXplain": _crosstab_dict(hx),
                      "IndicGuard": _crosstab_dict(ig)}

    train_pool, drop_a = _to_unified(pd.concat([macd, hx], ignore_index=True))
    external, drop_b = _to_unified(ig, extra=["category", "tag"])

    write_processed_csv(train_pool, UNIFIED_FILE)
    write_processed_csv(external, INDICGUARD_UNIFIED_FILE)

    summary = {
        "label_names": {str(k): v for k, v in LABEL_NAMES.items()},
        "verification": verification,
        "mapping_counts_raw_to_unified": mapping_counts,
        "rows_dropped": {"MACD+HateXplain": drop_a, "IndicGuard": drop_b},
        "files_written": {"train_pool": str(UNIFIED_FILE), "external": str(INDICGUARD_UNIFIED_FILE)},
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if verbose:
        print("\nUNIFIED TABLE (MACD + HateXplain)  ->", UNIFIED_FILE)
        tab = (train_pool.assign(cls=train_pool["label"].map(LABEL_NAMES))
               .pivot_table(index=["source", "language"], columns="cls", values="text",
                            aggfunc="count", fill_value=0))
        tab["total"] = tab.sum(axis=1)
        print(tab.to_string())
        print("\nEXTERNAL TABLE (IndicGuard, kept separate)  ->", INDICGUARD_UNIFIED_FILE)
        print(external["label"].map(LABEL_NAMES).value_counts().to_string())
        print(f"\nRows dropped: {drop_a}\n              {drop_b}")
        print("\nMapping evidence saved to results/metrics/label_mapping.json")
    return summary


if __name__ == "__main__":
    build_unified()
