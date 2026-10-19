"""PHASE 18 - Multilingual Evaluation & Testing Pipeline.

Performs:
1. Inspection and validation of the dedicated test manifest (data/evaluation/multilingual_test_manifest.csv).
2. Verification of available model artifacts (models/baseline/tfidf_lr.joblib and models/xlm_roberta_hate/).
3. Execution of inference if model checkpoints are present, adhering strictly to:
   - Zero fabrication of model predictions or confidences.
   - Distinct recording of predictions per model.
   - Exclusion of ambiguous / unadjudicated examples from supervised metrics.
4. Calculation and storage of testing metrics under results/metrics/.
5. Generation of results/predictions/test_predictions.csv.

Usage:
    python scripts/run_evaluation.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL_MANIFEST = ROOT / "data" / "evaluation" / "multilingual_test_manifest.csv"
PREDICTIONS_DIR = ROOT / "results" / "predictions"
METRICS_DIR = ROOT / "results" / "metrics"
BASELINE_PATH = ROOT / "models" / "baseline" / "tfidf_lr.joblib"
XLMR_DIR = ROOT / "models" / "xlm_roberta_hate"

PREDICTION_COLUMNS = [
    "text",
    "language",
    "category",
    "true_label",
    "predicted_label",
    "confidence",
    "is_correct",
    "model_name",
    "source",
    "is_synthetic",
    "review_status",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase18_evaluation")


def verify_manifest(manifest_path: Path = EVAL_MANIFEST) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    logger.info("Loaded evaluation manifest with %d samples from %s", len(df), manifest_path)

    required_cols = ["text", "language", "category", "true_label", "source", "is_synthetic", "review_status"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Manifest is missing required columns: {missing_cols}")

    # Check for empty text or duplicates
    if (df["text"].astype(str).str.strip() == "").any():
        raise ValueError("Manifest contains empty or whitespace-only texts!")
    if df["text"].duplicated().any():
        logger.warning("Manifest contains duplicate texts: %d duplicates", df["text"].duplicated().sum())

    return df


def check_model_artifacts() -> dict:
    status = {}

    # 1. Baseline TF-IDF + Logistic Regression
    baseline_exists = BASELINE_PATH.exists()
    status["baseline"] = {
        "name": "TF-IDF + Logistic Regression",
        "path": str(BASELINE_PATH),
        "available": baseline_exists,
        "detail": "Artifact found" if baseline_exists else "Artifact missing: models/baseline/tfidf_lr.joblib not found on disk"
    }

    # 2. XLM-RoBERTa
    xlmr_config = XLMR_DIR / "config.json"
    xlmr_weights = [XLMR_DIR / "model.safetensors", XLMR_DIR / "pytorch_model.bin"]
    has_weights = any(p.exists() for p in xlmr_weights)
    has_config = xlmr_config.exists()
    xlmr_available = has_weights and has_config

    status["xlm_roberta"] = {
        "name": "XLM-RoBERTa",
        "path": str(XLMR_DIR),
        "available": xlmr_available,
        "detail": "Artifacts found" if xlmr_available else (
            f"Artifact missing in {XLMR_DIR}: "
            f"config.json exists={has_config}, weights exists={has_weights} "
            "(Fine-tuned weights reside on Google Drive as documented in README.md)"
        )
    }

    # 3. IndicGuard Model check
    status["indicguard_model"] = {
        "name": "IndicGuard Model",
        "available": False,
        "detail": "IndicGuard is an evaluation dataset (L3Cube-Pune IndicGuard LLM safety corpus), not a trained model."
    }

    return status


def compute_supervised_metrics(y_true: list | np.ndarray, y_pred: list | np.ndarray, y_prob: list | np.ndarray | None = None) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            "non_hate": {
                "precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
                "support": int((y_true == 0).sum())
            },
            "hate_abusive": {
                "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "support": int((y_true == 1).sum())
            }
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }
    return metrics


def run_evaluation() -> dict:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_df = verify_manifest()
    model_status = check_model_artifacts()

    predictions_records = []
    evaluation_summary = {
        "manifest_path": str(EVAL_MANIFEST),
        "total_manifest_samples": len(manifest_df),
        "supervised_eligible_samples": int((manifest_df["review_status"] == "verified").sum()),
        "ambiguous_excluded_samples": int((manifest_df["review_status"] != "verified").sum()),
        "model_artifact_status": model_status,
        "models_evaluated": []
    }

    # Evaluate Baseline if available
    if model_status["baseline"]["available"]:
        logger.info("Loading baseline model from %s...", BASELINE_PATH)
        import joblib
        from src.data.preprocess import clean_for_tfidf
        baseline_bundle = joblib.load(BASELINE_PATH)
        vec = baseline_bundle["vectorizer"]
        model = baseline_bundle["model"]

        cleaned_texts = [clean_for_tfidf(t) for t in manifest_df["text"]]
        X = vec.transform(cleaned_texts)
        y_pred = model.predict(X)
        y_score = model.decision_function(X)
        # Convert decision function to pseudo-probability via sigmoid
        probs = 1.0 / (1.0 + np.exp(-y_score))

        for idx, row in manifest_df.iterrows():
            pred = int(y_pred[idx])
            conf = float(probs[idx] if pred == 1 else (1.0 - probs[idx]))
            true_lbl = row["true_label"]
            is_correct = bool(pred == int(true_lbl)) if pd.notna(true_lbl) else None

            predictions_records.append({
                "text": row["text"],
                "language": row["language"],
                "category": row["category"],
                "true_label": true_lbl if pd.notna(true_lbl) else None,
                "predicted_label": pred,
                "confidence": round(conf, 4),
                "is_correct": is_correct,
                "model_name": "TF-IDF + Logistic Regression",
                "source": row["source"],
                "is_synthetic": row["is_synthetic"],
                "review_status": row["review_status"]
            })
    else:
        logger.warning("Baseline model artifacts unavailable. Skipping baseline inference.")

    # Evaluate XLM-RoBERTa if available
    if model_status["xlm_roberta"]["available"]:
        logger.info("Loading XLM-RoBERTa model from %s...", XLMR_DIR)
        from src.model.evaluate import load_model, predict_probs
        from src.data.preprocess import clean_for_xlmr

        tokenizer, model, dev = load_model(XLMR_DIR)
        cleaned_texts = [clean_for_xlmr(t) for t in manifest_df["text"]]
        probs = predict_probs(model, tokenizer, cleaned_texts, show_progress=False)

        for idx, row in manifest_df.iterrows():
            p_non, p_hate = float(probs[idx, 0]), float(probs[idx, 1])
            pred = int(p_hate >= 0.5)
            conf = p_hate if pred == 1 else p_non
            true_lbl = row["true_label"]
            is_correct = bool(pred == int(true_lbl)) if pd.notna(true_lbl) else None

            predictions_records.append({
                "text": row["text"],
                "language": row["language"],
                "category": row["category"],
                "true_label": true_lbl if pd.notna(true_lbl) else None,
                "predicted_label": pred,
                "confidence": round(conf, 4),
                "is_correct": is_correct,
                "model_name": "XLM-RoBERTa",
                "source": row["source"],
                "is_synthetic": row["is_synthetic"],
                "review_status": row["review_status"]
            })
    else:
        logger.warning("XLM-RoBERTa model artifacts unavailable on local disk. Skipping XLM-R inference.")

    # Write predictions table
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    if predictions_records:
        preds_df = pd.DataFrame(predictions_records)
    else:
        # Schema-compliant empty table if models were unavailable on disk
        preds_df = pd.DataFrame(columns=PREDICTION_COLUMNS)

    preds_df.to_csv(pred_path, index=False, encoding="utf-8")
    logger.info("Saved predictions to %s (%d records)", pred_path, len(preds_df))

    # Calculate metrics if predictions exist
    if not preds_df.empty:
        for model_name, group in preds_df.groupby("model_name"):
            # Exclude ambiguous items
            eval_subset = group[group["review_status"] == "verified"]
            m = compute_supervised_metrics(
                eval_subset["true_label"].astype(int),
                eval_subset["predicted_label"].astype(int)
            )

            # Language-wise breakdown
            by_lang = {}
            for lang, lgroup in eval_subset.groupby("language"):
                if len(lgroup["true_label"].unique()) >= 2:
                    by_lang[lang] = compute_supervised_metrics(
                        lgroup["true_label"].astype(int),
                        lgroup["predicted_label"].astype(int)
                    )
                else:
                    by_lang[lang] = {
                        "sample_count": len(lgroup),
                        "note": "Single class present in subset; full confusion matrix/macro metrics deferred."
                    }

            evaluation_summary["models_evaluated"].append({
                "model_name": model_name,
                "overall": m,
                "by_language": by_lang
            })

    # Save Phase 18 manifest evaluation metrics
    metrics_path = METRICS_DIR / "phase18_manifest_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved Phase 18 metrics to %s", metrics_path)

    return evaluation_summary


if __name__ == "__main__":
    summary = run_evaluation()
    print("\n=== Phase 18 Evaluation Execution Summary ===")
    print(json.dumps(summary, indent=2))

