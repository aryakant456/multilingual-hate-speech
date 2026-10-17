"""Metric helpers shared by the baseline (Phase 7) and the XLM-R model (later phases).

Label 1 (Hate/Abusive) is the "positive" class, so precision / recall / F1 below are for
detecting hate. macro-F1 (average of both classes) is the main comparison number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.utils.config import LABEL_NAMES


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "n_hate": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_hate": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_hate": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_hate": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_non_hate": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def metrics_by_group(df: pd.DataFrame, group_col: str, y_pred, y_score=None) -> dict:
    """Metrics for every value of `group_col` (e.g. language or source). df needs a `label` column."""
    d = df.reset_index(drop=True).assign(_pred=np.asarray(y_pred))
    if y_score is not None:
        d["_score"] = np.asarray(y_score)
    res = {}
    for g, part in d.groupby(group_col):
        res[str(g)] = compute_metrics(part["label"], part["_pred"],
                                      part["_score"] if y_score is not None else None)
    return res


def plot_confusion_matrix(y_true, y_pred, path, title="Confusion matrix") -> None:
    """Save a two-panel figure (counts + row-normalised %). No pyplot -> safe in scripts and notebooks."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    fig = Figure(figsize=(10, 4.2))
    for ax, data, fmt, sub in zip(fig.subplots(1, 2), [cm, cm_pct], ["{:,.0f}", "{:.1f}%"],
                                  ["counts", "% of true class"]):
        ax.imshow(data, cmap="Blues")
        ax.set_xticks([0, 1], labels=names)
        ax.set_yticks([0, 1], labels=names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}\n({sub})", fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "black", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
