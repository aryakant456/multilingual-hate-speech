"""PHASE 20 - Visualization and Output Artifacts Generator.

Generates the 6 required visualization figures under results/figures/:
1. class_distribution.png     - Historical dataset class distribution across supported languages.
2. language_distribution.png  - Sample counts across train/val/test splits by language.
3. confusion_matrix.png       - Historical XLM-R benchmark confusion matrix (19,787 test samples).
4. language_f1.png            - Historical Baseline vs XLM-R Macro-F1 comparison across languages.
5. model_comparison.png       - Overall comparison of Baseline vs XLM-R on the official test set.
6. external_evaluation.png    - IndicGuard external evaluation (Broad vs Strict) across languages.

All figures use verified historical metric files in results/metrics/.
Zero synthetic data or fabricated predictions are used.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase20_visualizations")

# Set clean aesthetic style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})


def plot_class_distribution() -> Path:
    """1. class_distribution.png: Class breakdown across languages from results/metrics/class_distribution.csv."""
    src_file = METRICS_DIR / "class_distribution.csv"
    if not src_file.exists():
        raise FileNotFoundError(f"Missing source file: {src_file}")
    df = pd.read_csv(src_file)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    x = np.arange(len(df))
    width = 0.38

    bars1 = ax.bar(x - width/2, df["Non-Hate"], width, label="Non-Hate (0)", color="#2b5c8f", edgecolor="black", alpha=0.9)
    bars2 = ax.bar(x + width/2, df["Hate/Abusive"], width, label="Hate/Abusive (1)", color="#c93b2b", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Number of Comments")
    ax.set_title("Historical Dataset Class Distribution by Language\n(MACD + HateXplain Pool, n=171,651)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["language"])
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{int(h):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, rotation=0)

    plt.tight_layout()
    out_path = FIGURES_DIR / "class_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_language_distribution() -> Path:
    """2. language_distribution.png: Split breakdown by language from results/metrics/split_summary.csv."""
    src_file = METRICS_DIR / "split_summary.csv"
    if not src_file.exists():
        raise FileNotFoundError(f"Missing source file: {src_file}")
    df = pd.read_csv(src_file)

    # Pivot splits: train, val, test, zero_shot
    pivot_df = df.pivot_table(index="language", columns="set", values="total", aggfunc="sum", fill_value=0)
    # Order splits
    cols = [c for c in ["train", "val", "test", "zero_shot"] if c in pivot_df.columns]
    pivot_df = pivot_df[cols]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    colors = ["#2b5c8f", "#4f98ca", "#f29c38", "#a67bc5"]
    bottom = np.zeros(len(pivot_df))

    for idx, split in enumerate(cols):
        vals = pivot_df[split].values
        split_label = "Zero-shot Test" if split == "zero_shot" else split.capitalize()
        ax.bar(pivot_df.index, vals, bottom=bottom, label=f"{split_label} split", color=colors[idx % len(colors)], edgecolor="black", alpha=0.85)
        bottom += vals

    ax.set_ylabel("Total Number of Samples")
    ax.set_title("Dataset Distribution by Language & Split\n(Official Unified Datasets)")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, total in enumerate(bottom):
        ax.annotate(f"{int(total):,}",
                    xy=(i, total),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, weight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "language_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_confusion_matrix() -> Path:
    """3. confusion_matrix.png: XLM-R test confusion matrix from results/metrics/xlmr_test_metrics.json."""
    src_file = METRICS_DIR / "xlmr_test_metrics.json"
    if not src_file.exists():
        raise FileNotFoundError(f"Missing source file: {src_file}")

    with open(src_file, encoding="utf-8") as f:
        metrics = json.load(f)

    cm_data = metrics["test"]["overall"]["confusion_matrix"]
    tn, fp, fn, tp = cm_data["tn"], cm_data["fp"], cm_data["fn"], cm_data["tp"]
    cm = np.array([[tn, fp], [fn, tp]])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=150)
    labels = ["Non-Hate", "Hate/Abusive"]

    # Counts panel
    im1 = ax1.imshow(cm, cmap="Blues", interpolation="nearest")
    ax1.set_xticks([0, 1], labels=labels)
    ax1.set_yticks([0, 1], labels=labels)
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    ax1.set_title("Historical XLM-R Test Confusion Matrix\n(Sample Counts, n=19,787)")

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax1.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=12, weight="bold")

    # Percentage panel
    im2 = ax2.imshow(cm_pct, cmap="Blues", interpolation="nearest", vmin=0, vmax=100)
    ax2.set_xticks([0, 1], labels=labels)
    ax2.set_yticks([0, 1], labels=labels)
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    ax2.set_title("Historical XLM-R Test Confusion Matrix\n(Normalized Row %)")

    for i in range(2):
        for j in range(2):
            val = cm_pct[i, j]
            color = "white" if val > 50 else "black"
            ax2.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=12, weight="bold")

    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_language_f1() -> Path:
    """4. language_f1.png: Baseline vs XLM-R Macro-F1 across English, Hindi, Tamil, Malayalam."""
    src_file = METRICS_DIR / "model_comparison.csv"
    if not src_file.exists():
        raise FileNotFoundError(f"Missing source file: {src_file}")
    df = pd.read_csv(src_file)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width/2, df["Baseline macro-F1"], width, label="TF-IDF Baseline", color="#808b96", edgecolor="black", alpha=0.9)
    bars2 = ax.bar(x + width/2, df["XLM-R macro-F1"], width, label="XLM-RoBERTa (Fine-Tuned)", color="#2e86c1", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Macro-F1 Score")
    ax.set_ylim(0.65, 0.95)
    ax.set_title("Historical Model Performance Across Languages\n(Held-out Test Split, Macro-F1 Comparison)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Language"])
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, weight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "language_f1.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_model_comparison() -> Path:
    """5. model_comparison.png: Overall comparison of Baseline vs XLM-R across multiple metrics."""
    baseline_file = METRICS_DIR / "baseline_metrics.json"
    xlmr_file = METRICS_DIR / "xlmr_test_metrics.json"

    with open(baseline_file, encoding="utf-8") as f:
        b_data = json.load(f)["test"]["overall"]
    with open(xlmr_file, encoding="utf-8") as f:
        x_data = json.load(f)["test"]["overall"]

    metrics_names = ["Accuracy", "Macro-F1", "Weighted-F1", "Hate F1", "Non-Hate F1", "ROC-AUC"]
    b_scores = [b_data["accuracy"], b_data["macro_f1"], b_data["weighted_f1"], b_data["f1_hate"], b_data["f1_non_hate"], b_data["roc_auc"]]
    x_scores = [x_data["accuracy"], x_data["macro_f1"], x_data["weighted_f1"], x_data["f1_hate"], x_data["f1_non_hate"], x_data["roc_auc"]]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, b_scores, width, label="TF-IDF + Logistic Regression", color="#7f8c8d", edgecolor="black", alpha=0.9)
    bars2 = ax.bar(x + width/2, x_scores, width, label="XLM-RoBERTa (Fine-Tuned)", color="#27ae60", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Score")
    ax.set_ylim(0.70, 1.00)
    ax.set_title("Overall Historical Benchmark Comparison: Baseline vs XLM-RoBERTa\n(Official Held-out Test Set, n=19,787)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, weight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_external_evaluation() -> Path:
    """6. external_evaluation.png: IndicGuard Broad vs Strict evaluation across languages."""
    src_file = METRICS_DIR / "xlmr_internal_vs_external.csv"
    if not src_file.exists():
        raise FileNotFoundError(f"Missing source file: {src_file}")
    df = pd.read_csv(src_file)

    # Filter to IndicGuard rows
    ig_broad = df[df["Dataset"] == "IndicGuard (broad)"].set_index("Language")
    ig_strict = df[df["Dataset"] == "IndicGuard (strict)"].set_index("Language")
    internal = df[df["Dataset"] == "Internal"].set_index("Language")

    languages = ["English", "Hindi", "Tamil", "Malayalam"]
    x = np.arange(len(languages))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=150)

    bars_int = ax.bar(x - width, [internal.loc[l, "Macro-F1"] for l in languages], width,
                      label="In-Domain Internal Test (n=19,787)", color="#2e86c1", edgecolor="black", alpha=0.85)
    bars_broad = ax.bar(x, [ig_broad.loc[l, "Macro-F1"] for l in languages], width,
                        label="IndicGuard Broad External (n=77,677)", color="#e67e22", edgecolor="black", alpha=0.85)
    bars_strict = ax.bar(x + width, [ig_strict.loc[l, "Macro-F1"] for l in languages], width,
                         label="IndicGuard Strict External (n=59,709)", color="#27ae60", edgecolor="black", alpha=0.85)

    ax.set_ylabel("Macro-F1 Score")
    ax.set_ylim(0.35, 1.02)
    ax.set_title("In-Domain vs External IndicGuard Benchmark Performance (XLM-R)\n(Evidence of Domain Shift from Social Media to LLM Safety Prompts)")
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in list(bars_int) + list(bars_broad) + list(bars_strict):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, weight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "external_evaluation.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def generate_all_visualizations() -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_class_distribution(),
        plot_language_distribution(),
        plot_confusion_matrix(),
        plot_language_f1(),
        plot_model_comparison(),
        plot_external_evaluation(),
    ]
    return paths


if __name__ == "__main__":
    generated = generate_all_visualizations()
    print("\n=== Phase 20 Generated Visualizations ===")
    for p in generated:
        print(f"  {p.name:<26} ({p.stat().st_size:,} bytes) -> {p}")

