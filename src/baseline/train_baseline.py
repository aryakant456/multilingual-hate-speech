"""PHASE 7 - Baseline: TF-IDF + Logistic Regression.

Run from the project root (after Phase 5):
    python -m src.baseline.train_baseline            # full small grid search
    python -m src.baseline.train_baseline --quick    # one setting, for a fast test

Pipeline:   text_tfidf  ->  TF-IDF (unigrams or unigrams+bigrams)  ->  Logistic Regression
  * settings are chosen on the VALIDATION set only (macro-F1)
  * the test set is used once, for the final numbers
  * class_weight="balanced" so the hate / non-hate imbalance does not bias the model
  * MACD Telugu/Kannada (zero-shot) are scored at the end; the model never trains on them
IndicGuard is NOT used here - it is reserved for the final external evaluation.

Writes: results/metrics/baseline_metrics.json, results/metrics/baseline_val_grid.csv,
        results/figures/baseline_confusion_matrix.png, results/predictions/baseline_predictions.csv,
        models/baseline/tfidf_lr.joblib
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.io_utils import read_processed_csv
from src.data.preprocess import whitespace_tokenize
from src.utils.config import (BASELINE_DIR, FIGURES_DIR, METRICS_DIR,
                              PREDICTIONS_DIR, RANDOM_SEED, TEST_FILE,
                              TRAIN_FILE, VAL_FILE, ZERO_SHOT_FILE)
from src.utils.metrics import compute_metrics, metrics_by_group, plot_confusion_matrix

NGRAM_RANGES = [(1, 1), (1, 2)]        # unigrams  vs  unigrams + bigrams
C_VALUES = [1.0, 2.0, 4.0, 8.0, 16.0]  # strength of regularisation (bigger C = weaker)


def make_vectorizer(ngram_range) -> TfidfVectorizer:
    return TfidfVectorizer(tokenizer=whitespace_tokenize, token_pattern=None, lowercase=False,
                           ngram_range=ngram_range, min_df=2, sublinear_tf=True,
                           max_features=500_000, dtype=np.float32)


def make_model(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                              max_iter=1000, random_state=RANDOM_SEED)


def top_features(vectorizer, model, n: int = 15) -> dict:
    """Most indicative n-grams for each class (for the report; shows real abusive words!)."""
    names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(model.coef_[0])
    return {"non_hate": names[order[:n]].tolist(), "hate": names[order[::-1][:n]].tolist()}


def run_baseline(quick: bool = False, verbose: bool = True) -> dict:
    for p in (TRAIN_FILE, VAL_FILE, TEST_FILE, ZERO_SHOT_FILE):
        if not p.exists():
            raise SystemExit(f"{p.name} not found - run Phases 3-5 first.")
    train, val, test, zero = (read_processed_csv(p) for p in (TRAIN_FILE, VAL_FILE, TEST_FILE, ZERO_SHOT_FILE))
    for d in (train, val, test, zero):               # a text can be empty after tfidf cleaning (emoji-only etc. are kept)
        d["text_tfidf"] = d["text_tfidf"].fillna("")
    if verbose:
        print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} | zero-shot {len(zero):,}")

    # ---- 1. grid search on the validation set --------------------------------------
    grid, best = [], None
    t0 = time.time()
    for ngr in ([(1, 2)] if quick else NGRAM_RANGES):
        vec = make_vectorizer(ngr)
        X_tr = vec.fit_transform(train["text_tfidf"])            # fitted on TRAIN only
        X_va = vec.transform(val["text_tfidf"])
        for C in ([1.0] if quick else C_VALUES):
            model = make_model(C).fit(X_tr, train["label"])
            m = compute_metrics(val["label"], model.predict(X_va), model.decision_function(X_va))
            grid.append({"ngram_range": str(ngr), "C": C, "val_macro_f1": m["macro_f1"],
                         "val_f1_hate": m["f1_hate"], "val_accuracy": m["accuracy"], "n_features": X_tr.shape[1]})
            if verbose:
                print(f"  ngram={ngr} C={C:<4} val macro-F1={m['macro_f1']:.4f}  hate-F1={m['f1_hate']:.4f}  "
                      f"({X_tr.shape[1]:,} features)")
            if best is None or m["macro_f1"] > best["score"]:
                best = {"score": m["macro_f1"], "ngram_range": ngr, "C": C, "vec": vec, "model": model}

    vec, model = best["vec"], best["model"]
    cfg = {"ngram_range": list(best["ngram_range"]), "C": best["C"], "class_weight": "balanced",
           "selected_on": "validation macro-F1"}
    if verbose:
        print(f"\nBest on validation: {cfg}   (grid took {time.time() - t0:.0f}s)")

    # ---- 2. final evaluation -------------------------------------------------------
    results = {"model": "TF-IDF + Logistic Regression", "best_config": cfg, "quick_mode": quick,
               "train_size": int(len(train))}
    preds = []
    for name, d in [("validation", val), ("test", test), ("zero_shot", zero)]:
        X = vec.transform(d["text_tfidf"])
        y_pred, y_score = model.predict(X), model.decision_function(X)
        entry = {"overall": compute_metrics(d["label"], y_pred, y_score),
                 "by_language": metrics_by_group(d, "language", y_pred, y_score)}
        if name != "zero_shot":
            entry["by_source"] = metrics_by_group(d, "source", y_pred, y_score)
        results[name] = entry
        if name != "validation":
            preds.append(d[["text", "label", "language", "source"]].assign(set=name, pred=y_pred, score=y_score))
        if name == "test":
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            plot_confusion_matrix(d["label"], y_pred, FIGURES_DIR / "baseline_confusion_matrix.png",
                                  "TF-IDF + LR - test set")

    # ---- 3. save everything ----------------------------------------------------------
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(grid).to_csv(METRICS_DIR / "baseline_val_grid.csv", index=False)
    pd.concat(preds).to_csv(PREDICTIONS_DIR / "baseline_predictions.csv", index=False, encoding="utf-8")
    joblib.dump({"vectorizer": vec, "model": model, "config": cfg}, BASELINE_DIR / "tfidf_lr.joblib")

    if verbose:
        t, z = results["test"]["overall"], results["zero_shot"]["overall"]
        print(f"\nTEST      accuracy={t['accuracy']:.4f}  macro-F1={t['macro_f1']:.4f}  hate-F1={t['f1_hate']:.4f}  AUC={t['roc_auc']:.4f}")
        for lang, m in results["test"]["by_language"].items():
            print(f"   {lang:<10} macro-F1={m['macro_f1']:.4f}  n={m['n']:,}")
        print(f"ZERO-SHOT macro-F1={z['macro_f1']:.4f}  (languages never seen in training)")
        print("\nSaved: results/metrics/baseline_metrics.json, baseline_val_grid.csv, "
              "results/figures/baseline_confusion_matrix.png, models/baseline/tfidf_lr.joblib")
    results["_objects"] = {"vectorizer": vec, "model": model}   # for the notebook only (not saved)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="single setting instead of the full grid")
    run_baseline(quick=ap.parse_args().quick)
