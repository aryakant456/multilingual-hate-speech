# Multilingual Hate Speech Detection — Project Completion Plan (Phases 18–21)

## Overview & Architecture Understanding
- **Task**: Multilingual Binary Hate & Abusive Speech Detection (0 = Non-Hate / Non-Abusive, 1 = Hate / Abusive).
- **Supported Languages**: English (`en`), Hindi (`hi`), Tamil (`ta`), Malayalam (`ml`), plus zero-shot evaluation on Telugu (`te`) and Kannada (`kn`).
- **Primary Datasets**:
  - MACD (Multi-lingual Abusive Comment Dataset for Indic languages: Hindi, Tamil, Malayalam, Telugu, Kannada)
  - HateXplain (English benchmark with normal, offensive, and hatespeech labels)
  - IndicGuard (External LLM safety evaluation benchmark from L3Cube-Pune)
- **Trained Models & Baselines**:
  - Baseline: TF-IDF (unigram + bigram) + Logistic Regression (C=4.0, class_weight='balanced')
  - Primary Model: XLM-RoBERTa (`xlm-roberta-base`, sequence classification fine-tuned on train split)
  - Explainability: Integrated Gradients (LayerIntegratedGradients on embeddings) + Occlusion sensitivity fallback
- **Current Status**:
  - Phases 1–17 completed: Pipeline code, EDA, baseline metrics, XLM-R Colab training/eval metrics, FastAPI service, and latency benchmarks are established.
  - Official test split size: 19,787 samples (English: 1,918, Hindi: 6,723, Tamil: 5,992, Malayalam: 5,154).
  - Official test results:
    - Baseline TF-IDF + Logistic Regression: Accuracy = 82.80%, Macro-F1 = 82.80%
    - XLM-RoBERTa: Accuracy = 86.12%, Macro-F1 = 86.11%
    - XLM-RoBERTa on IndicGuard (external zero-shot domain): Accuracy = 84.88%, Macro-F1 = 55.91%

---

## Dataset Acquisition & Environment Verification (Step 1 Verification Status)

- [x] **HateXplain Download**: Complete and verified on disk.
  - `data/raw/hatexplain/dataset.json` (12.26 MB)
  - `data/raw/hatexplain/post_id_divisions.json` (591.9 KB)
- [x] **MACD Acquisition**: Complete and verified on disk.
  - Repository cloned into `data/raw/macd/`.
  - `data/raw/macd/dataset/` contains all 15 official CSV files (5 languages x train/val/test).
  - 152,422 total rows (74,550 abusive / 77,872 non-abusive), perfectly matching the NeurIPS 2022 paper.
- [x] **IndicGuard Snapshot**: Complete and verified on disk.
  - Contains language folders (`English`, `Hindi`, `Tamil`, `Malyalam`, `Bengali`, `Kannada`, `Marathi`, `Punjabi`, `Telugu`, `Urdu`).
  - Total raw archives: `all-lang-train-test-valid.zip` (454.8 MB) and `additional-test-sets.zip` (8.3 MB).
- [x] **Python Environment & Dependencies**:
  - `seaborn` (v0.13.2) and `pyarrow` (v26.0.0) successfully installed and importable.
  - Full machine learning and NLP stack verified (`torch` 2.12.0+cpu, `transformers` 5.9.0, `scikit-learn` 1.8.0, `pandas` 3.0.3, `numpy` 2.4.6, `matplotlib` 3.10.9).
  - Environment check `scripts/check_phase_1_2.py`: **25 PASS, 2 WARN, 0 FAIL**.
- [x] **Dedicated Evaluation Dataset Prepared**:
  - Located at `data/evaluation/multilingual_test_manifest.csv`.
  - 42 samples across all required categories (Normal English, Hateful English, Ambiguous English, Normal Hindi, Hateful Hindi, Normal Tamil, Hateful Tamil, Normal Malayalam, Hateful Malayalam, Hinglish, Tanglish, Manglish).
  - Data integrity checks passed: 0 missing text, 0 duplicate rows, 38 verified binary labels (19 Hate, 19 Non-Hate), 4 ambiguous samples marked `needs_adjudication` with null true_label.

---

## Phase Checklist (Phases 18–21)

### Phase 18: Multilingual Testing
- [x] **Step 18.1**: Inspect data pipeline, schema, mappings, and model specifications.
  - Documented text columns (`text`), label column (`label` binary), language names, and class mapping (0 = Non-Hate, 1 = Hate/Abusive).
- [x] **Step 18.2**: Create a dedicated evaluation dataset under `data/evaluation/multilingual_test_manifest.csv`.
  - Verified 42 curated samples covering English, Hindi, Tamil, Malayalam, and code-mixed pairs (38 verified, 4 ambiguous).
- [x] **Step 18.3**: Run / record inference pipeline via `scripts/run_evaluation.py` into `results/predictions/test_predictions.csv`.
  - Verified model artifact status (`models/baseline/tfidf_lr.joblib` and `models/xlm_roberta_hate/`). Preserved clean schema without fabricated outputs.
- [x] **Step 18.4**: Calculate testing metrics and report findings in `results/PHASE_18_TESTING_REPORT.md` and `results/metrics/phase18_manifest_metrics.json`.
  - Comprehensive reporting covering the 42-sample manifest, the 19,787-sample official benchmark test set, and IndicGuard external evaluation.

### Phase 19: Error Analysis
- [x] **Step 19.1**: Error analysis pipeline & historical benchmark error extraction.
  - Implemented and unit-tested `scripts/analyze_errors.py`.
  - Initialized schema-compliant output files: `results/error_analysis/false_positives.csv` and `results/error_analysis/false_negatives.csv`.
  - Traced and analyzed historical benchmark errors (XLM-R: FP 1,390, FN 1,357; Baseline: FP 1,568, FN 1,833; IndicGuard: FP 1,586, FN 10,158).
- [x] **Step 19.2**: Linguistic error typology, heuristics & report generation.
  - Implemented heuristic indicators for sarcasm, slang, code-mixing, implicit hate, spelling variation, and negation/quotes with `automated_flag_needs_human_verification` tags.
  - Generated comprehensive reports: `results/error_analysis/error_analysis_report.md` and `results/PHASE_19_ERROR_ANALYSIS_REPORT.md`.
  - **Prepared-but-Unverified (Blocked)**: Live per-example error extraction on the 42-sample manifest is blocked pending restoration of local model weights from Google Drive (`MyDrive/hate-speech/xlm_roberta_hate`). All 42 manifest rows remain untouched.

### Phase 20: Results and Visualizations
- [x] Ensure structured output directory format: `results/figures/`, `results/metrics/`, `results/predictions/`, `results/error_analysis/`.
- [x] Verify / generate required visualizations:
  - `results/figures/class_distribution.png` (Class distribution across languages, n=171,651)
  - `results/figures/language_distribution.png` (Split breakdown by language: train, val, test, zero-shot)
  - `results/figures/confusion_matrix.png` (Historical XLM-R test confusion matrix counts & row %)
  - `results/figures/language_f1.png` (Baseline vs XLM-R Macro-F1 across English, Hindi, Tamil, Malayalam)
  - `results/figures/model_comparison.png` (Multi-metric comparison: Baseline vs XLM-R on test split)
  - `results/figures/external_evaluation.png` (In-domain test vs IndicGuard Broad vs Strict across languages)
- [x] Verify / generate compliant metrics JSON files:
  - `results/metrics/baseline_metrics.json` (Verified historical baseline metrics)
  - `results/metrics/xlm_metrics.json` (Standardized XLM-R test metrics)
  - `results/metrics/indicguard_metrics.json` (Standardized IndicGuard external metrics)
- [x] Generate comprehensive report at `results/PHASE_20_VISUALIZATION_REPORT.md`.

### Phase 21: Final Model Comparison and Academic Report
- [x] **Step 21.1**: Combined test comparison table generated at `results/metrics/final_model_comparison.csv`.
  - Contrasts Baseline TF-IDF (82.80% Acc, 0.8280 Macro-F1) vs XLM-R (86.12% Acc, 0.8611 Macro-F1) vs IndicGuard External (84.88% Acc, 0.5591 Macro-F1).
- [x] **Step 21.2**: Language-wise comparison table generated at `results/metrics/final_language_comparison.csv`.
  - Detailed sample counts, baseline vs XLM-R Macro-F1, gains, and error rates for English, Hindi, Tamil, Malayalam, and Combined.
- [x] **Step 21.3**: Final academic reports generated:
  - `results/PHASE_21_FINAL_COMPARISON_REPORT.md`
  - `results/PROJECT_FINAL_SUMMARY.md`
  - `results/FINAL_RESULTS_REPORT.md` (covering all 16 academic sections)
- [x] **End-to-End Pipeline & Artifact Validation**: Verified all generated CSVs, PNGs, and JSONs for non-empty content, mathematical consistency, and reproducible scripts.
- [ ] **Pending Teammate Action**: Fresh per-example evaluation on the 42-row manifest and per-example error extraction remain pending restoration of model weights from Google Drive (`MyDrive/hate-speech/xlm_roberta_hate`). All 42 manifest rows remain intact.
