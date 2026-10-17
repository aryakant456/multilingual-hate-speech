"""PHASES 10-11 - Evaluate the TRAINED XLM-R model (internal test + IndicGuard external test).

Only inference happens here: the model is loaded from disk, put in eval mode, its parameters are
frozen, and a SHA-256 fingerprint of the weights file is recorded so the results can prove that
the model was NOT changed between the internal test and the external IndicGuard test.

The functions are split in two groups:
  * torch functions   : load_model, predict_probs          (need torch + transformers)
  * pure-pandas tools : evaluate_frame, language_table, strict_external, comparison tables, plots
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.utils.config import (EXTERNAL_LANGUAGES, INTERNAL_LANGUAGES, XLMR_DIR)
from src.utils.metrics import compute_metrics, metrics_by_group


# ------------------------------------------------------------------ torch functions
def model_fingerprint(model_dir=XLMR_DIR) -> str:
    """SHA-256 of the weights file. If it is identical before and after the external test,
    the model was not modified."""
    d = Path(model_dir)
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (d / name).exists():
            path = d / name
            break
    else:
        raise FileNotFoundError(f"No weights file (model.safetensors) in {d}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(model_dir=XLMR_DIR, device: str | None = None):
    """Load the fine-tuned model for inference only. Returns (tokenizer, model, device)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"No trained model in {model_dir}. Train it on Colab (notebook 04) and download the folder "
            "into models/xlm_roberta_hate/ .")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)          # "locked": cannot be trained by accident
    return tokenizer, model, dev


def predict_probs(model, tokenizer, texts, max_length: int = 128, batch_size: int = 128,
                  show_progress: bool = True) -> np.ndarray:
    """Softmax probabilities, shape [n, 2] = [P(NON-HATE), P(HATE)], in the ORIGINAL order of `texts`.
    Texts are processed sorted by length (much faster padding) and put back in order afterwards."""
    import torch
    from tqdm.auto import tqdm
    dev = next(model.parameters()).device
    texts = [("" if t is None else str(t)) for t in texts]
    order = np.argsort([len(t) for t in texts], kind="stable")
    probs = np.zeros((len(texts), 2), dtype=np.float32)
    use_amp = dev.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.bfloat16
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), disable=not show_progress, desc="predicting"):
            idx = order[start:start + batch_size]
            enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                            pad_to_multiple_of=8, return_tensors="pt").to(dev)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(**enc).logits
            probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs


# ------------------------------------------------------------------ pure pandas / numpy tools
def evaluate_frame(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.5,
                   group_cols=("language",)) -> tuple[dict, np.ndarray]:
    """Metrics overall and per group. df needs `label` (+ the group columns). Threshold is FIXED (0.5):
    it must never be tuned on test / external data."""
    y = df["label"].to_numpy()
    score = probs[:, 1]
    pred = (score >= threshold).astype(int)
    out = {"threshold": threshold, "overall": compute_metrics(y, pred, score)}
    for g in group_cols:
        if g in df.columns:
            out[f"by_{g}"] = metrics_by_group(df, g, pred, score)
    return out, pred


def language_table(by_language: dict, languages=INTERNAL_LANGUAGES) -> pd.DataFrame:
    """Language | Precision | Recall | F1 (all for the HATE class) | Macro-F1 | Accuracy | n"""
    rows = []
    for lg in languages:
        m = by_language.get(lg)
        if m is None:
            continue
        rows.append({"Language": lg, "Precision": m["precision_hate"], "Recall": m["recall_hate"],
                     "F1": m["f1_hate"], "Macro-F1": m["macro_f1"], "Accuracy": m["accuracy"], "n": m["n"]})
    return pd.DataFrame(rows)


def normalise_tag(tag) -> str:
    """'Jail breaking' / 'Jailbreaking' / 'jailbreaking' -> 'jailbreaking' (IndicGuard spells tags inconsistently)."""
    if tag is None or (isinstance(tag, float) and tag != tag) or pd.isna(tag):
        return ""
    return "".join(ch for ch in str(tag).lower() if ch.isalpha())


def strict_external(df: pd.DataFrame) -> pd.DataFrame:
    """STRICT IndicGuard version: remove jailbreak / persona prompts, which are about tricking a chatbot,
    not about hateful comments. (The BROAD version keeps every row.)"""
    if "tag" not in df.columns:
        raise KeyError("column 'tag' missing - re-run the data pipeline (Phases 3-5)")
    return df[~df["tag"].map(normalise_tag).str.contains("jail")].reset_index(drop=True)


def comparison_rows(dataset_name: str, by_language: dict, languages) -> list[dict]:
    rows = []
    for lg in languages:
        m = by_language.get(lg)
        if m is None:
            continue
        rows.append({"Dataset": dataset_name, "Language": lg, "Macro-F1": m["macro_f1"], "F1 (hate)": m["f1_hate"],
                     "Precision (hate)": m["precision_hate"], "Recall (hate)": m["recall_hate"],
                     "ROC-AUC": m.get("roc_auc"), "PR-AUC": m.get("pr_auc"), "n": m["n"],
                     "hate_%": round(100 * m["n_hate"] / m["n"], 1)})
    return rows


def baseline_vs_xlmr(baseline_json: dict, xlmr_by_language: dict, languages=INTERNAL_LANGUAGES) -> pd.DataFrame:
    """Test-set macro-F1 and hate-F1 of the TF-IDF baseline next to XLM-R, per language."""
    rows = []
    for lg in languages:
        b = baseline_json["test"]["by_language"].get(lg)
        x = xlmr_by_language.get(lg)
        if b is None or x is None:
            continue
        rows.append({"Language": lg, "Baseline macro-F1": b["macro_f1"], "XLM-R macro-F1": x["macro_f1"],
                     "Gain": x["macro_f1"] - b["macro_f1"], "Baseline F1 (hate)": b["f1_hate"],
                     "XLM-R F1 (hate)": x["f1_hate"]})
    return pd.DataFrame(rows)


def plot_internal_vs_external(table: pd.DataFrame, path, metric: str = "Macro-F1") -> None:
    """Grouped bars: one group per language, one bar per dataset."""
    langs = list(dict.fromkeys(table["Language"]))
    datasets = list(dict.fromkeys(table["Dataset"]))
    fig = Figure(figsize=(9, 4.2))
    ax = fig.subplots()
    width = 0.8 / max(len(datasets), 1)
    for i, ds in enumerate(datasets):
        vals = [table[(table["Dataset"] == ds) & (table["Language"] == lg)][metric].mean() for lg in langs]
        bars = ax.bar(np.arange(len(langs)) + i * width, vals, width, label=ds)
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(langs)) + width * (len(datasets) - 1) / 2, labels=langs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: internal test vs IndicGuard external test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
