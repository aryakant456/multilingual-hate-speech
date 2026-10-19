# FINAL RESULTS REPORT: Multilingual Hate Speech Detection Project

*(This comprehensive report mirrors and complements `results/PROJECT_FINAL_SUMMARY.md` and `results/PHASE_21_FINAL_COMPARISON_REPORT.md` in accordance with the Phase 21 academic report specifications).*

## 1. Project Overview & Architecture
An NLP-based deep transfer learning and baseline framework for binary hate and abusive speech detection across **English, Hindi, Tamil, and Malayalam**, with zero-shot transfer evaluated on **Kannada and Telugu**.

- **Baseline**: Character/word TF-IDF ($n$-gram range 1–2, $C=4.0$) + Logistic Regression with balanced class weighting.
- **Deep Model**: Fine-tuned **XLM-RoBERTa** (`xlm-roberta-base`, 278M parameters, 12 layers).
- **Explainability**: Integrated Gradients attribution with Occlusion sensitivity fallback.
- **Service Layer**: FastAPI asynchronous microservice with latency benchmarking.

---

## 2. Dataset Description
- **MACD**: 152,422 social comments (Hindi, Tamil, Malayalam, Kannada, Telugu).
- **HateXplain**: 20,148 English social posts with 3-annotator majority vote.
- **IndicGuard**: 431,464 prompts across 11 Indic languages (external LLM safety test set).

---

## 3. Label Definitions & Unified Mapping
Unified binary target:
- **`0`**: Non-Hate / Non-Abusive
- **`1`**: Hate / Abusive

### Mapping Rules:
- MACD: Raw `0` $\rightarrow$ `1` (Abusive); Raw `1` $\rightarrow$ `0` (Non-Abusive).
- HateXplain: `hatespeech` $\rightarrow$ `1`, `offensive` $\rightarrow$ `1`, `normal` $\rightarrow$ `0`, `undecided` $\rightarrow$ dropped.
- IndicGuard: `violated_categories` containing `Hate/Identity Hate` $\rightarrow$ `1`, `safe` $\rightarrow$ `0`, other unsafe $\rightarrow$ dropped.

---

## 4. Evaluation Methodology
- **Internal Held-Out Test Set**: 19,787 samples (English: 1,918, Hindi: 6,723, Malayalam: 5,154, Tamil: 5,992).
- **Zero-Shot Test Set**: 62,857 samples (Kannada: 32,902, Telugu: 29,955).
- **External Evaluation**: 77,677 samples on IndicGuard (Broad: 77,677, Strict: 59,709).
- **Dedicated Manifest**: 42 curated samples (38 verified, 4 ambiguous).

---

## 5. Models Evaluated
- Baseline: TF-IDF + Logistic Regression
- Primary: Fine-tuned XLM-RoBERTa

---

## 6. Overall Metrics (Official Held-Out Test Split, $n=19,787$)
| Metric | TF-IDF Baseline | XLM-RoBERTa (Fine-Tuned) | Absolute Improvement |
|---|---|---|---|
| **Accuracy** | 82.80% | **86.12%** | **+3.32%** |
| **Macro-F1** | 0.8280 | **0.8611** | **+0.0331** |
| **Weighted-F1** | 0.8281 | **0.8612** | **+0.0331** |
| **Hate F1 ($F_{1, \text{hate}}$)** | 0.8230 | **0.8592** | **+0.0362** |
| **Non-Hate F1 ($F_{1, \text{non-hate}}$)** | 0.8330 | **0.8631** | **+0.0301** |
| **Macro Precision** | 0.8281 | **0.8611** | **+0.0330** |
| **Macro Recall** | 0.8277 | **0.8612** | **+0.0335** |
| **ROC-AUC** | 0.9091 | **0.9399** | **+0.0308** |

---

## 7. Language-Wise Performance
| Language | Samples ($n$) | Baseline Macro-F1 | XLM-R Macro-F1 | XLM-R Accuracy | XLM-R Hate F1 | Macro-F1 Gain |
|---|---|---|---|---|---|---|
| **English** | 1,918 | 0.7455 | **0.7836** | 79.25% | 0.8274 | **+3.81%** |
| **Hindi** | 6,723 | 0.8255 | **0.8628** | 86.30% | 0.8677 | **+3.73%** |
| **Tamil** | 5,992 | 0.8461 | **0.8788** | 87.93% | 0.8710 | **+3.27%** |
| **Malayalam** | 5,154 | 0.8325 | **0.8618** | 86.32% | 0.8478 | **+2.93%** |
| **Combined ($ALL$)** | 19,787 | 0.8280 | **0.8611** | 86.12% | 0.8592 | **+3.32%** |

---

## 8. Testing Category & Code-Mixed Findings
- Tamil achieved the highest Macro-F1 (0.8788) and lowest error rates.
- English has lower Macro-F1 (0.7836) due to nuances of slur reclamation, sarcasm, and conversational discourse in HateXplain.
- Code-mixed categories (Hinglish, Tanglish, Manglish) in the 42-row manifest are un-evaluated locally, pending checkpoint restoration.

---

## 9. Confusion Matrix Analysis
- XLM-R Test Matrix ($n=19,787$):
  - True Negatives: 8,660 (86.2% of true non-hate)
  - False Positives: 1,390 (13.8% of true non-hate)
  - False Negatives: 1,357 (13.9% of true hate)
  - True Positives: 8,380 (86.1% of true hate)

---

## 10. False-Positive and False-Negative Analysis
- **Overall FPR**: 13.83% (Baseline: 15.60%)
- **Overall FNR**: 13.94% (Baseline: 18.83%)
- XLM-RoBERTa achieved a **26.0% reduction in false negatives** compared to the baseline, significantly improving detection safety.

---

## 11. Error Categories and Examples
Typology heuristics implemented in `scripts/analyze_errors.py`:
1. Sarcasm / Irony
2. Slang & Colloquial Pejoratives
3. Code-Mixing & Script Blending
4. Implicit Hate
5. Context Dependency / Reclamation
6. Spelling Variations / Elongation
7. Negation & Quoted Speech

All heuristic tags are marked `review_status = "automated_flag_needs_human_verification"`. Template files exist in `results/error_analysis/` awaiting live model restoration.

---

## 12. Baseline vs. Transformer Comparison
XLM-RoBERTa consistently outperforms the TF-IDF baseline across all 4 languages, offering contextual token embeddings that capture sentence semantics rather than isolated bag-of-words.

---

## 13. External Evaluation (IndicGuard)
- **Broad** ($n=77,677$): Macro-F1 = 0.5591, Accuracy = 84.88%, Hate F1 = 0.2017, FNR = 87.25%
- **Strict** ($n=59,709$): Macro-F1 = 0.5843, Accuracy = 85.19%, Hate F1 = 0.2393, FNR = 84.34%
- **Key Insight**: Severe domain shift between conversational social media comments and translated synthetic LLM prompts.

---

## 14. Explainability & Latency Profiles
- Attribution via Integrated Gradients on word embeddings with occlusion sensitivity fallback.
- Median prediction latency: 39.5 ms. Explanation overhead: ~1.95 s for positive hate comments.

---

## 15. Limitations
- Content sensitivity: Genuine abusive language in social media data.
- Contextual ambiguity in reclaimed slurs.
- Domain sensitivity across differing data genres.

---

## 16. Teammate Checkpoint Restoration Protocol & Future Work
To enable live inference on the 42-row manifest:
1. Copy `xlm_roberta_hate/` from Google Drive (`/content/drive/MyDrive/hate-speech/xlm_roberta_hate`, SHA-256: `fda10ac6bf81d189ab599a080cae5c159e05325a374c09797ac9245c8fb5f396`) to local `models/xlm_roberta_hate/`.
2. Run:
   ```bash
   python scripts/run_evaluation.py
   python scripts/analyze_errors.py
   ```
This immediately extracts live per-example false positives and false negatives without retraining.

