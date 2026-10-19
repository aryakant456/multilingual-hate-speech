# Phase 20 — Visualizations and Output Artifacts Report

## 1. Executive Summary & Overview
Phase 20 synthesizes experimental and linguistic findings into standard visualization artifacts under `results/figures/`.

All 6 required visualizations have been generated, validated against historical benchmark metric files, and verified for non-empty binary output. In compliance with strict project integrity guidelines:
- All figures are derived exclusively from verified historical metric files in `results/metrics/`.
- Zero synthetic data, fabricated predictions, or placeholder numbers were used.
- Historical benchmark test results (19,787 samples) and IndicGuard external evaluation results (77,677 samples) are clearly distinguished from the 42-row demonstration manifest.
- The 42-row manifest is explicitly recognized as awaiting local model checkpoint restoration.

---

## 2. Inventory of Generated Figures & Data Provenance

| Figure Filename | Artifact Path | File Size | Primary Source Data | Key Metrics Visualized |
|---|---|---|---|---|
| **`class_distribution.png`** | `results/figures/class_distribution.png` | 60,662 B | `results/metrics/class_distribution.csv` | Class distribution (Non-Hate vs Hate) across English, Hindi, Kannada, Malayalam, Tamil, Telugu ($n=171,651$). |
| **`language_distribution.png`** | `results/figures/language_distribution.png` | 59,359 B | `results/metrics/split_summary.csv` | Split partitions (Train: 68,997, Val: 19,788, Test: 19,787, Zero-shot: 62,857) across languages. |
| **`confusion_matrix.png`** | `results/figures/confusion_matrix.png` | 66,042 B | `results/metrics/xlmr_test_metrics.json` | Historical XLM-R test confusion matrix: Raw counts ($n=19,787$) and normalized row percentages. |
| **`language_f1.png`** | `results/figures/language_f1.png` | 60,149 B | `results/metrics/model_comparison.csv` | Baseline vs XLM-R Macro-F1 across English, Hindi, Tamil, Malayalam, and Overall ($ALL$). |
| **`model_comparison.png`** | `results/figures/model_comparison.png` | 62,790 B | `baseline_metrics.json`, `xlmr_test_metrics.json` | Multimetric comparison (Accuracy, Macro-F1, Weighted-F1, Hate F1, Non-Hate F1, ROC-AUC) on the held-out test split. |
| **`external_evaluation.png`** | `results/figures/external_evaluation.png` | 67,988 B | `results/metrics/xlmr_internal_vs_external.csv` | Generalization comparison: Internal In-Domain Test ($n=19,787$) vs IndicGuard Broad ($n=77,677$) vs IndicGuard Strict ($n=59,709$). |

---

## 3. Important Findings Supported by the Plots

1. **Transformer Superiority Across All Languages (`language_f1.png`, `model_comparison.png`)**:
   - XLM-RoBERTa consistently outperforms the TF-IDF + Logistic Regression baseline across all tested languages:
     - English: Macro-F1 increases from 0.7455 to **0.7836** (+3.81% gain).
     - Hindi: Macro-F1 increases from 0.8255 to **0.8628** (+3.73% gain).
     - Tamil: Macro-F1 increases from 0.8461 to **0.8788** (+3.27% gain).
     - Malayalam: Macro-F1 increases from 0.8325 to **0.8618** (+2.93% gain).
     - Overall ($n=19,787$): Macro-F1 rises from 0.8280 to **0.8611**, Accuracy from 82.80% to **86.12%**, and ROC-AUC reaches **0.9399**.
2. **Symmetric Error Profile on Internal Test (`confusion_matrix.png`)**:
   - On the official balanced held-out test set ($n=19,787$), XLM-RoBERTa achieves balanced precision and recall:
     - True Negatives: 8,660 (86.2% of non-hate)
     - False Positives: 1,390 (13.8% of non-hate)
     - False Negatives: 1,357 (13.9% of hate)
     - True Positives: 8,380 (86.1% of hate)
3. **Severe Domain Shift on LLM Safety Benchmarks (`external_evaluation.png`)**:
   - In-domain test performance (Macro-F1: 0.78–0.88) drops dramatically when evaluated on IndicGuard (Macro-F1: 0.49–0.63).
   - This drop occurs because IndicGuard consists of machine-translated adversarial prompts and synthetic jailbreaks rather than organic social media comments.
   - Removing adversarial jailbreak prompts (IndicGuard Strict, $n=59,709$) recovers performance slightly across all four languages (+1.0% to +3.6% Macro-F1 improvement over Broad).
4. **Data Composition and Language Scale (`class_distribution.png`, `language_distribution.png`)**:
   - The unified dataset maintains a relatively balanced class distribution (Hate share: English 59.4%, Hindi 52.0%, Kannada 48.3%, Malayalam 45.0%, Tamil 46.4%, Telugu 52.0%).
   - Zero-shot languages (Kannada: 32,902, Telugu: 29,955) represent substantial un-fine-tuned evaluation potential.

---

## 4. Data Validation & Mathematical Checks

All plotted figures were audited against their original source data:
- **Confusion Matrix Sum Check**:
  $$\text{TN} + \text{FP} + \text{FN} + \text{TP} = 8,660 + 1,390 + 1,357 + 8,380 = 19,787$$
  This matches the official test set size in `results/metrics/split_summary.csv` ($1,918 + 6,723 + 5,154 + 5,992 = 19,787$).
- **Class Sum Check**:
  $$\text{Non-Hate Support} = \text{TN} + \text{FP} = 8,660 + 1,390 = 10,050$$
  $$\text{Hate Support} = \text{FN} + \text{TP} = 1,357 + 8,380 = 9,737$$
  $$10,050 + 9,737 = 19,787$$
  Matching the ground-truth distribution in `split_summary.csv` (Non-Hate: $777 + 3,229 + 2,846 + 3,198 = 10,050$; Hate: $1,141 + 3,494 + 2,308 + 2,794 = 9,737$).
- **Metric Range Verification**: All F1 scores, accuracies, and percentages fall within $[0.0, 1.0]$ and $[0\%, 100\%]$.

---

## 5. Missing Data Limitations & Status of Blocked Figures

- **No Figures Blocked**: All 6 required figures are generated and verified.
- **Dedicated 42-Row Manifest Distinction**:
  - The 42-row manifest in `data/evaluation/multilingual_test_manifest.csv` was **not** plotted as evaluated data because local forward inference is pending model weight restoration.
  - The plots correctly present the official benchmark test set ($n=19,787$) and IndicGuard ($n=77,677$), clearly indicating their historical benchmark provenance.

