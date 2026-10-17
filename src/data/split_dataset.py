"""PHASE 5 - Dataset splitting (leakage-safe).

Run from the project root (after Phases 3 and 4):
    python -m src.data.split_dataset

Design
  train / val / test : MACD Hindi + Tamil + Malayalam  and  HateXplain English.
                       The datasets' OFFICIAL splits are used as they are - the official
                       test sets are never re-shuffled.
  zero_shot_test     : MACD Telugu + Kannada. Never used for training or tuning.
  external test      : IndicGuard, written to its own file and never mixed with the above.

Safety checks (the script stops or removes rows if a check fails)
  * no text appears in two different sets (train / val / test / zero-shot)
  * IndicGuard prompts that also occur in any MACD/HateXplain set are removed
  * hate share per language is compared between train and val/test (stratification check)
A stratified fallback (80/10/10 by language x label) is used ONLY for rows whose
dataset has no official split.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.data.io_utils import read_processed_csv, write_processed_csv
from src.utils.config import (CLEAN_FILE, EXTERNAL_FILE, INDICGUARD_CLEAN_FILE,
                              LABEL_NAMES, METRICS_DIR, RANDOM_SEED, TEST_FILE,
                              TRAIN_FILE, TRAIN_MACD_LANGUAGES, VAL_FILE,
                              ZERO_SHOT_FILE, ZERO_SHOT_LANGUAGES)

OUT_COLUMNS = ["text", "text_tfidf", "label", "language", "source", "split", "label_raw"]


def stratified_fallback(sub: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.Series:
    """80/10/10 split stratified by language x label. Returns a Series of 'train'/'val'/'test'."""
    strata = sub["language"] + "_" + sub["label"].astype(str)
    train_idx, rest_idx = train_test_split(sub.index, test_size=0.2, stratify=strata, random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, stratify=strata.loc[rest_idx], random_state=seed)
    out = pd.Series(index=sub.index, dtype=object)
    out.loc[train_idx], out.loc[val_idx], out.loc[test_idx] = "train", "val", "test"
    return out


def class_weights_for(y: pd.Series) -> dict:
    """'balanced' weights: n_samples / (n_classes * n_class_samples). Used later as the loss weights."""
    classes = np.array(sorted(y.unique()))
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {str(c): float(x) for c, x in zip(classes, w)}


def main(verbose: bool = True) -> dict:
    if not CLEAN_FILE.exists():
        raise SystemExit("Run Phases 3 and 4 first:  python -m src.data.combine_datasets  &&  python -m src.data.preprocess")
    df = read_processed_csv(CLEAN_FILE)
    notes: list[str] = []

    # ---- 1. decide which rows belong where -------------------------------------
    is_zero = (df["source"] == "MACD") & df["language"].isin(ZERO_SHOT_LANGUAGES)
    is_pool = ((df["source"] == "MACD") & df["language"].isin(TRAIN_MACD_LANGUAGES)) | (df["source"] == "HateXplain")
    unused = df.loc[~is_zero & ~is_pool]
    if len(unused):
        notes.append(f"{len(unused)} rows with languages {sorted(unused['language'].unique())} are in no set (not used).")

    pool = df.loc[is_pool].copy()
    zero = df.loc[is_zero].copy()
    zero["split"] = "zero_shot"

    unknown = ~pool["split"].isin(["train", "val", "test"])
    if unknown.any():
        pool.loc[unknown, "split"] = stratified_fallback(pool.loc[unknown])
        notes.append(f"{int(unknown.sum())} rows had no official split -> stratified 80/10/10 fallback.")

    train, val, test = (pool[pool["split"] == s] for s in ["train", "val", "test"])

    # ---- 2. leakage check: the same text in two sets? ----------------------------
    sets = {"train": train, "val": val, "test": test, "zero_shot": zero}
    keys = {k: set(v["text"].str.casefold()) for k, v in sets.items()}
    overlaps = {}
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a}&{b}"] = len(keys[a] & keys[b])
    if any(overlaps.values()):
        # remove the overlapping rows from the lower-priority set (train > val > test > zero_shot)
        for a, b in [("train", "val"), ("train", "test"), ("val", "test"),
                     ("train", "zero_shot"), ("val", "zero_shot"), ("test", "zero_shot")]:
            bad = keys[a] & keys[b]
            if bad:
                sets[b] = sets[b][~sets[b]["text"].str.casefold().isin(bad)]
                keys[b] = set(sets[b]["text"].str.casefold())
                notes.append(f"removed {len(bad)} texts from {b} that also occur in {a}.")
        train, val, test, zero = sets["train"], sets["val"], sets["test"], sets["zero_shot"]

    # ---- 3. IndicGuard stays separate; drop prompts that also occur anywhere above ----
    ext = read_processed_csv(INDICGUARD_CLEAN_FILE)
    seen = set().union(*[set(v["text"].str.casefold()) for v in (train, val, test, zero)])
    leak = ext["text"].str.casefold().isin(seen)
    notes.append(f"IndicGuard: removed {int(leak.sum())} prompts that also occur in the other datasets.")
    ext = ext.loc[~leak].reset_index(drop=True)

    # ---- 4. write the files -------------------------------------------------------
    for part, path in [(train, TRAIN_FILE), (val, VAL_FILE), (test, TEST_FILE), (zero, ZERO_SHOT_FILE)]:
        write_processed_csv(part[OUT_COLUMNS].reset_index(drop=True), path)
    write_processed_csv(ext, EXTERNAL_FILE)

    # ---- 5. report: sizes, stratification check, class weights ----------------------
    def share(d):  # hate share per language
        return d.groupby("language")["label"].mean()
    tr_share = share(train)
    strat_flags = []
    for nm, d in [("val", val), ("test", test)]:
        diff = (share(d) - tr_share).abs().dropna()
        for lang, x in diff.items():
            if x > 0.03:
                strat_flags.append(f"{nm}/{lang}: hate share differs from train by {100 * x:.1f} points")

    report = {
        "sizes": {k: int(len(v)) for k, v in [("train", train), ("val", val), ("test", test),
                                             ("zero_shot", zero), ("external_indicguard", ext)]},
        "overlaps_before_cleanup": overlaps,
        "notes": notes,
        "stratification_flags": strat_flags,
        "class_weights_train": class_weights_for(train["label"]),
        "train_hate_share_by_language": {k: round(float(v), 4) for k, v in tr_share.items()},
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    summary = pd.concat({"train": train, "val": val, "test": test, "zero_shot": zero}, names=["set"]).reset_index(level=0)
    summary["class"] = summary["label"].map(LABEL_NAMES)
    tab = summary.pivot_table(index=["set", "source", "language"], columns="class", values="text",
                              aggfunc="count", fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab["hate_%"] = (100 * tab["Hate/Abusive"] / tab["total"]).round(1)
    tab.to_csv(METRICS_DIR / "split_summary.csv")

    if verbose:
        print(tab.to_string())
        print("\nSizes:", report["sizes"])
        print("Class weights (train, for the weighted loss):", report["class_weights_train"])
        print("Overlaps found before clean-up:", overlaps)
        for n in notes:
            print(" note:", n)
        print("Stratification flags:", strat_flags or "none (hate share per language is similar in all splits)")
        print(f"\nFiles written to {TRAIN_FILE.parent} and {EXTERNAL_FILE.parent}")
    return report


if __name__ == "__main__":
    main()
