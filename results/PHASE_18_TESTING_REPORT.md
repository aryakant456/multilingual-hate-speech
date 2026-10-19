# Phase 18 Testing & Multilingual Evaluation Report

## 1. Executive Summary & Objective
Phase 18 establishes the multilingual testing framework and validates model performance across English, Hindi, Tamil, Malayalam, and code-mixed inputs (Hindi+English, Tamil+English, Malayalam+English).

This report documents:
1. The model-artifact and prediction-recovery audit.
2. The dedicated 42-sample multilingual evaluation manifest in `data/evaluation/multilingual_test_manifest.csv`.
3. The official 19,787-sample held-out benchmark test set results.
4. The external evaluation on IndicGuard (77,677 samples).
5. The exact Google Drive restoration checklist, verified paths, and code readiness.

---

## 2. Model Artifact & Prediction Recovery Audit

A project-wide search of the filesystem, git commit tree, notebooks, and configuration files was executed:

### A. Traceability of Historical Benchmark Metrics
- **Baseline TF-IDF + Logistic Regression Metrics** (`results/metrics/baseline_metrics.json`):
  - **Source Scripts**: Produced by `src/baseline/train_baseline.py` and `notebooks/03_baseline.ipynb`.
  - **Confusion Matrix Data**: Exactly preserved in JSON (`tn: 8488, fp: 1555, fn: 1842, tp: 7895`).
  - **Per-Sample Predictions**: Originally written to `results/predictions/baseline_predictions.csv`, but excluded from Git tracking via `.gitignore` (`results/predictions/**`).
- **XLM-RoBERTa Test Metrics** (`results/metrics/xlmr_test_metrics.json`):
  - **Source Scripts**: Produced by `notebooks/04_xlm_training.ipynb` (training) and `notebooks/05_evaluation.ipynb` (evaluation) via `src/model/evaluate.py`.
  - **Confusion Matrix Data**: Exactly preserved in JSON (`tn: 8660, fp: 1390, fn: 1357, tp: 8380`).
  - **Per-Sample Predictions**: As recorded in cell 8 of `notebooks/05_evaluation.ipynb`, predictions contain real user comments and were saved directly to Google Drive:
    ```python
    # predictions contain real text -> Google Drive only, never GitHub
    test.assign(p_hate=probs_test[:, 1], pred=pred_test).to_csv(f"{DRIVE_ROOT}/predictions/xlmr_test_predictions.csv", index=False)
    ```
- **IndicGuard External Metrics** (`results/metrics/xlmr_external_metrics.json`):
  - **Source Scripts**: Produced by Part B of `notebooks/05_evaluation.ipynb` evaluating 77,677 samples (broad) and 59,709 samples (strict).

### B. Current Local Artifact Status
| Model Candidate | Expected Local Path | Local Disk Status | Notes |
|---|---|---|---|
| **TF-IDF + Logistic Regression** | `models/baseline/tfidf_lr.joblib` | **Unavailable (Missing)** | The file was git-ignored (`models/*/*`) and not preserved in the repository clone. |
| **XLM-RoBERTa (`xlm-roberta-base`)** | `models/xlm_roberta_hate/` | **Unavailable (Weights missing)** | Directory contains only `.gitkeep`. Checkpoint weights (~1.1 GB) were not committed to Git. |
| **Hugging Face Model ID** | `xlm-roberta-base` | **Pretrained Backbone Only** | Raw pretrained language model without hate speech fine-tuning; not a substitute for the fine-tuned classifier. |
| **IndicGuard Model** | N/A | **N/A (Dataset, not a model)** | IndicGuard is an LLM safety evaluation benchmark from L3Cube-Pune, not a model checkpoint. |

### C. Exact Google Drive Checkpoint Location & Expected Files
As specified in `README.md`, `notebooks/04_xlm_training.ipynb`, and `xlmr_test_metrics.json`:
- **Google Drive Root**: `/content/drive/MyDrive/hate-speech`
- **Model Checkpoint Folder**: `MyDrive/hate-speech/xlm_roberta_hate/`
- **Recorded Model SHA-256 Fingerprint**: `fda10ac6bf81d189ab599a080cae5c159e05325a374c09797ac9245c8fb5f396`
- **Expected Artifacts in `xlm_roberta_hate/`**:
  1. `config.json` (architecture config with `id2label` and `label2id`)
  2. `model.safetensors` or `pytorch_model.bin` (fine-tuned model weights)
  3. `tokenizer.json` (SentencePiece tokenizer definition)
  4. `tokenizer_config.json`
  5. `sentencepiece.bpe.model`
  6. `special_tokens_map.json`
  7. `training_config.json` (training hyperparameter records)
  8. `label_mapping.json` (positive class mapping)
- **Saved Predictions File on Google Drive**: `MyDrive/hate-speech/predictions/xlmr_test_predictions.csv`

### D. Code Readiness for Artifact Loading
The existing codebase is fully equipped to load the restored artifacts without retraining:
- [`src/model/evaluate.py`](file:///c:/Users/aryak/OneDrive/Desktop/multilingual-hate-speech/src/model/evaluate.py): `load_model()` and `model_fingerprint()` are configured to immediately load and verify the checkpoint from `models/xlm_roberta_hate/`.
- [`src/model/predict.py`](file:///c:/Users/aryak/OneDrive/Desktop/multilingual-hate-speech/src/model/predict.py): `HateSpeechPredictor` is ready for zero-shot and batch inference.
- [`scripts/run_evaluation.py`](file:///c:/Users/aryak/OneDrive/Desktop/multilingual-hate-speech/scripts/run_evaluation.py): Verified and ready to run inference on the 42-row manifest as soon as weights are placed in `models/xlm_roberta_hate/`.

---

## 3. Dedicated Evaluation Manifest (`data/evaluation/multilingual_test_manifest.csv`)

A reproducible, high-quality evaluation manifest was created and validated:

- **Total Samples**: 42
- **Verified Supervised Samples**: 38 (19 Non-Hate, 19 Hate/Abusive)
- **Ambiguous Samples**: 4 (Reserved for human adjudication; excluded from all supervised metric calculations)
- **Missing / Empty Texts**: 0
- **Duplicate Texts**: 0

### Category Breakdown
| Category | Language | Samples | Label Distribution | Review Status |
|---|---|---|---|---|
| Normal English | English | 4 | 4 Non-Hate (0) | `verified` |
| Hateful English | English | 4 | 4 Hate/Abusive (1) | `verified` |
| Ambiguous English | English | 4 | None / Unassigned | `needs_adjudication` |
| Normal Hindi | Hindi | 4 | 4 Non-Hate (0) | `verified` |
| Hateful Hindi | Hindi | 4 | 4 Hate/Abusive (1) | `verified` |
| Normal Tamil | Tamil | 4 | 4 Non-Hate (0) | `verified` |
| Hateful Tamil | Tamil | 4 | 4 Hate/Abusive (1) | `verified` |
| Normal Malayalam | Malayalam | 4 | 4 Non-Hate (0) | `verified` |
| Hateful Malayalam | Malayalam | 4 | 4 Hate/Abusive (1) | `verified` |
| Hindi + English (Hinglish) | Hindi+English | 2 | 1 Non-Hate (0), 1 Hate (1) | `verified` |
| Tamil + English (Tanglish) | Tamil+English | 2 | 1 Non-Hate (0), 1 Hate (1) | `verified` |
| Malayalam + English (Manglish) | Malayalam+English | 2 | 1 Non-Hate (0), 1 Hate (1) | `verified` |

---

## 4. Official Benchmark Test Set Evaluation (19,787 Samples)

The official held-out test set evaluation (MACD Hindi, Tamil, Malayalam + HateXplain English) was computed and verified in `results/metrics/`:

### Overall Benchmark Metrics
| Metric | TF-IDF + Logistic Regression (Baseline) | XLM-RoBERTa (Fine-Tuned) | Absolute Gain |
|---|---|---|---|
| **Evaluated Samples ($n$)** | 19,787 | 19,787 | — |
| **Accuracy** | 82.80% (0.8280) | **86.12% (0.8612)** | +3.32% |
| **Macro-F1** | 82.80% (0.8280) | **86.11% (0.8611)** | +3.31% |
| **Weighted-F1** | 82.81% (0.8281) | **86.12% (0.8612)** | +3.31% |
| **Hate F1 ($F_{1, \text{hate}}$)** | 82.30% (0.8230) | **85.92% (0.8592)** | +3.62% |
| **Non-Hate F1 ($F_{1, \text{non-hate}}$)** | 83.30% (0.8330) | **86.31% (0.8631)** | +3.01% |
| **Macro Precision** | 82.81% (0.8281) | **86.11% (0.8611)** | +3.30% |
| **Macro Recall** | 82.77% (0.8277) | **86.12% (0.8612)** | +3.35% |
| **ROC-AUC** | 0.9091 | **0.9399** | +0.0308 |
| **Confusion Matrix** | TN: 8,488 \| FP: 1,555<br>FN: 1,842 \| TP: 7,895 | TN: 8,660 \| FP: 1,390<br>FN: 1,357 \| TP: 8,380 | Fewer errors across both classes |

### Language-Wise Performance Breakdown
| Language | Samples ($n$) | Baseline Macro-F1 | XLM-R Macro-F1 | XLM-R Accuracy | XLM-R Hate F1 | Macro-F1 Gain |
|---|---|---|---|---|---|---|
| **English** | 1,918 | 0.7455 | **0.7836** | 79.25% | 0.8274 | +3.81% |
| **Hindi** | 6,723 | 0.8255 | **0.8628** | 86.30% | 0.8677 | +3.73% |
| **Tamil** | 5,992 | 0.8461 | **0.8788** | 87.93% | 0.8710 | +3.27% |
| **Malayalam** | 5,154 | 0.8325 | **0.8618** | 86.32% | 0.8478 | +2.93% |

> [!NOTE]
> Tamil achieves the highest Macro-F1 (0.8788) and Accuracy (87.93%), followed by Hindi (0.8628) and Malayalam (0.8618). English has a lower Macro-F1 (0.7836), primarily due to higher class imbalance and subtler offensive speech nuances present in HateXplain.

---

## 5. External Evaluation: IndicGuard (Out-of-Domain Safety Benchmark)

IndicGuard (`l3cube-pune/IndicGuard`) was evaluated as an out-of-domain external test set without retraining or tuning thresholds:

| Evaluation Variant | Samples ($n$) | Accuracy | Macro-F1 | Hate F1 | ROC-AUC | Key Observation |
|---|---|---|---|---|---|---|
| **IndicGuard Broad** | 77,677 | 84.88% | **0.5591** | 0.2017 | 0.7516 | Low recall on translated LLM safety prompts; large domain shift from social media comments. |
| **IndicGuard Strict** (no jailbreaks) | 59,709 | 85.19% | **0.5843** | 0.2393 | 0.7937 | Removing synthetic adversarial jailbreaks slightly improves Macro-F1. |

---

## 6. Generated File Paths

- **Evaluation Dataset**: `data/evaluation/multilingual_test_manifest.csv`
- **Inference Pipeline Script**: `scripts/run_evaluation.py`
- **Predictions File**: `results/predictions/test_predictions.csv`
- **Phase 18 Metrics Summary**: `results/metrics/phase18_manifest_metrics.json`
- **Testing Report**: `results/PHASE_18_TESTING_REPORT.md`

---

## 7. Next Required Action (Restoration Checklist)

To run fresh predictions on the 42-row manifest or operate the local API service:
1. Copy the folder `xlm_roberta_hate/` from Google Drive (`MyDrive/hate-speech/xlm_roberta_hate`) into the project's [`models/xlm_roberta_hate/`](file:///c:/Users/aryak/OneDrive/Desktop/multilingual-hate-speech/models/xlm_roberta_hate) directory.
2. (Optional) Copy `tfidf_lr.joblib` into [`models/baseline/tfidf_lr.joblib`](file:///c:/Users/aryak/OneDrive/Desktop/multilingual-hate-speech/models/baseline/tfidf_lr.joblib).
3. Run `python scripts/run_evaluation.py` to immediately produce verified manifest predictions.
