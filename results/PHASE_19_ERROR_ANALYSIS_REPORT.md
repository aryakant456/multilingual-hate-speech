# Phase 19 — Multilingual Error Analysis Report

## 1. Executive Summary & Objective
Phase 19 investigates model error modes across multilingual hate and abusive speech detection tasks. It aims to isolate false positives (FP: Non-Hate predicted as Hate) and false negatives (FN: Hate predicted as Non-Hate), assess failure typologies (sarcasm, slang, code-mixing, implicit hate, spelling variations), and quantify error rates across English, Hindi, Tamil, Malayalam, and code-mixed inputs.

In adherence to project rules:
- **No synthetic predictions were fabricated.**
- **No unverified example-level false positives or negatives were invented from aggregate numbers.**
- Historical benchmark confusion matrices from the official test set (19,787 samples) and IndicGuard external set (77,677 samples) are analyzed and reported as historical results.
- The 42-row multilingual evaluation manifest remains untouched.

---

## 2. Task Status Categorization

| Task | Status | Details |
|---|---|---|
| **Prediction Data Search & Audit** | **Completed** | Inspected `results/predictions/`, `results/metrics/`, and `notebooks/05_evaluation.ipynb`. Confirmed that per-sample prediction CSVs were excluded by `.gitignore` and reside on Google Drive. |
| **Error Analysis Pipeline Implementation** | **Completed & Verified** | Built and unit-tested `scripts/analyze_errors.py` with schema validation, confusion matrix extraction, and heuristic categorization. |
| **Historical Benchmark Error Analysis** | **Completed** | Compiled exact FP, FN, TN, TP counts, FPR, and FNR across all supported languages from official metric JSONs. |
| **Local Fresh Manifest Error Analysis** | **Prepared-but-Unverified (Blocked)** | `scripts/analyze_errors.py` created schema-compliant `results/error_analysis/false_positives.csv` and `false_negatives.csv` (0 rows). Per-sample error extraction is blocked until model checkpoints are restored. |

---

## 3. Historical Benchmark Error Analysis (19,787 Samples)

The official held-out test set (MACD Hindi, Tamil, Malayalam + HateXplain English) provides the empirical baseline for error analysis:

### A. Overall Error Rates
| Model | Total Evaluated ($n$) | False Positives (FP) | False Negatives (FN) | False Positive Rate (FPR) | False Negative Rate (FNR) |
|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | 19,787 | 1,568 | 1,833 | **15.60%** | **18.83%** |
| **XLM-RoBERTa (Fine-Tuned)** | 19,787 | 1,390 | 1,357 | **13.83%** | **13.94%** |
| **Relative Improvement** | — | -178 (-11.4%) | -476 (-26.0%) | -1.77% | -4.89% |

> [!NOTE]
> Fine-tuned XLM-RoBERTa achieves a substantial reduction in both error types, particularly reducing false negatives by 26.0%. This indicates that contextual embeddings better capture subtle abusive vocabulary compared to n-gram surface forms.

### B. Language-Wise Error Breakdown (XLM-RoBERTa)
| Language | Samples ($n$) | Non-Hate Support | Hate Support | False Positives (FP) | False Negatives (FN) | FPR (%) | FNR (%) |
|---|---|---|---|---|---|---|---|
| **English** | 1,918 | 777 | 1,141 | 211 | 187 | **27.16%** | **16.39%** |
| **Hindi** | 6,723 | 3,229 | 3,494 | 448 | 473 | **13.87%** | **13.54%** |
| **Tamil** | 5,992 | 3,198 | 2,794 | 370 | 353 | **11.57%** | **12.63%** |
| **Malayalam** | 5,154 | 2,846 | 2,308 | 361 | 344 | **12.68%** | **14.90%** |

#### Key Linguistic Observations:
1. **English Higher FPR (27.16%)**: English (from HateXplain) has a noticeably higher false-positive rate than Indic languages. HateXplain contains conversational Twitter/Reddit posts discussing controversial topics, identity terms, and reclaimed slurs, causing the model to over-predict hate when offensive words occur without hateful intent.
2. **Indic Languages Consistency**: Hindi, Tamil, and Malayalam exhibit balanced error profiles with FPR and FNR tightly clustered between 11.5% and 14.9%.
3. **Tamil Lowest Error Rate**: Tamil achieves the lowest FPR (11.57%) and lowest FNR (12.63%), correlating with its highest Macro-F1 (0.8788).

### C. External Benchmark Error Rates (IndicGuard, 77,677 Samples)
- **False Positives (FP)**: 1,586 (FPR: **2.40%**)
- **False Negatives (FN)**: 10,158 (FNR: **87.25%**)
- **Analysis**: The extreme false negative rate (87.25%) on IndicGuard underscores the domain mismatch between social media abusive text and machine-translated LLM safety evaluation prompts. While the model rarely flags benign prompts as hate (low FPR), it fails to detect synthetic jailbreaks that do not resemble conversational social abuse.

---

## 4. Error Categorization & Linguistic Typology

The error-analysis pipeline (`scripts/analyze_errors.py`) implements transparent heuristic rules for classifying failure mechanisms when per-sample predictions are provided:

1. **Sarcasm & Irony**: Inverted sentiment where polite or laudatory vocabulary masks hostile intent.
2. **Slang & Colloquial Pejoratives**: Regional abusive terms (e.g. Hindi *saale*, Malayalam *myre*, Tamil *thendi*) with sparse representations.
3. **Code-Mixing & Script Blending**: Mixed Latin and Indic script sequences (Hinglish, Tanglish, Manglish) challenging standard subword tokenization.
4. **Implicit Hate**: Toxic intent expressed without explicit dictionary profanities.
5. **Context Dependency & Slur Reclamation**: Identity-affirming reclamation of slurs or quotation in news/academic contexts.
6. **Spelling Variations & Character Lengthening**: Deliberate lengthening (e.g. *fiiiiire*, *hateeee*) obscuring token boundaries.
7. **Negation & Quoted Speech**: Inverted polarity where a user negates or quotes a slur to condemn it.

> [!IMPORTANT]
> All automated classifications assigned by `scripts/analyze_errors.py` are explicitly marked with `review_status = "automated_flag_needs_human_verification"`. Automated heuristics are transparent indicators and must not be presented as adjudicated human ground truth.

---

## 5. Artifacts Created & Output Files

- **Pipeline Script**: `scripts/analyze_errors.py` (validated with unit tests).
- **False Positives File**: `results/error_analysis/false_positives.csv` (schema initialized).
- **False Negatives File**: `results/error_analysis/false_negatives.csv` (schema initialized).
- **Report File**: `results/error_analysis/error_analysis_report.md`.
- **Phase 19 Academic Report**: `results/PHASE_19_ERROR_ANALYSIS_REPORT.md`.

---

## 6. Remaining Blockers & Next Steps

1. **Checkpoint Restoration**: Live extraction of per-example false positives and false negatives from the 42-row dedicated manifest requires downloading the model folder `xlm_roberta_hate/` from Google Drive (`MyDrive/hate-speech/xlm_roberta_hate/`) into `models/xlm_roberta_hate/`.
2. **Automated Execution**: Once restored, running:
   ```bash
   python scripts/run_evaluation.py
   python scripts/analyze_errors.py
   ```
   will immediately populate `false_positives.csv` and `false_negatives.csv` with per-sample text, linguistic heuristic tags, and model confidence scores.

