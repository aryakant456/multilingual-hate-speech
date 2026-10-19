# Multilingual NLP Hate & Abusive Speech Detection — Project Final Summary

## 1. Project Overview & Architecture
This project develops an end-to-end multilingual Natural Language Processing (NLP) framework for detecting hate and abusive speech across four key Indian and global languages: **English, Hindi, Tamil, and Malayalam**, alongside zero-shot evaluation on **Kannada and Telugu**. 

The system implements a dual-tier modeling approach:
1. **N-gram Baseline**: Word/character TF-IDF vectorization paired with regularized Logistic Regression.
2. **Deep Multilingual Transformer**: Sequence classification fine-tuning of **XLM-RoBERTa** (`xlm-roberta-base`), leveraging cross-lingual pretraining across 100 languages.
3. **Model Explainability**: Word-level attribution via **Integrated Gradients** (Captum LayerIntegratedGradients) with an occlusion sensitivity fallback.
4. **Service Infrastructure**: High-performance asynchronous **FastAPI** backend with a responsive **React (Vite)** moderation UI.

---

## 2. Dataset Descriptions
The framework unifies three distinct data corpora:
- **MACD (Multi-lingual Abusive Comment Dataset)**: 152,422 real-world social media comments across Hindi, Tamil, Malayalam, Kannada, and Telugu (NeurIPS 2022). Balanced class distribution (74,550 abusive / 77,872 non-abusive).
- **HateXplain**: 20,148 English social posts (Twitter and Gab) annotated by 3 annotators with majority vote labels (`hatespeech`, `offensive`, `normal`).
- **IndicGuard (L3Cube-Pune)**: 431,464 prompts across 11 Indic languages, serving as an out-of-domain external LLM safety benchmark.

---

## 3. Unified Label Definitions & Normalization
To bridge heterogeneous source taxonomies, a unified binary schema was established:
- **`0` = NON-HATE / NON-ABUSIVE**: Normal, benign, non-toxic text.
- **`1` = HATE / ABUSIVE**: Targeted hate speech, abusive comments, and severe harassment.

### Dataset Mapping Rules:
- **MACD**: Raw `0` (abusive in original paper) $\rightarrow$ mapped to `1`. Raw `1` (non-abusive) $\rightarrow$ mapped to `0`. (Inverted deliberately to align with positive class conventions).
- **HateXplain**: `hatespeech` $\rightarrow$ `1`; `offensive` $\rightarrow$ `1`; `normal` $\rightarrow$ `0`; `undecided` (annotators disagreed) $\rightarrow$ dropped.
- **IndicGuard**: `violated_categories` containing `Hate/Identity Hate` $\rightarrow$ `1`; `prompt_label == safe` $\rightarrow$ `0`; other unsafe prompts (e.g. criminal planning) $\rightarrow$ dropped.

---

## 4. Evaluation Methodology
- **In-Domain Internal Split**: Official held-out partitions were preserved without random reshuffling:
  - **Train**: 68,997 samples (English: 15,350, Hindi: 20,167, Malayalam: 15,492, Tamil: 17,988).
  - **Validation**: 19,788 samples.
  - **Test**: 19,787 samples (English: 1,918, Hindi: 6,723, Malayalam: 5,154, Tamil: 5,992).
- **Zero-Shot Transfer**: MACD Kannada (32,902 samples) and Telugu (29,955 samples) evaluated without prior language-specific training.
- **External Evaluation**: IndicGuard Broad (77,677 samples) and Strict (59,709 samples) evaluated with frozen model weights and fixed 0.5 classification threshold.
- **Dedicated Manifest**: 42 curated samples covering English, Hindi, Tamil, Malayalam, and code-mixed pairs (38 verified, 4 ambiguous).

---

## 5. Models Evaluated
1. **TF-IDF + Logistic Regression (Baseline)**:
   - Features: Word unigrams + bigrams, sublinear term frequency, min document frequency 2, max 500,000 features.
   - Classifier: Logistic Regression with $C=4.0$, `class_weight='balanced'`, `liblinear` solver.
2. **XLM-RoBERTa (Transformer)**:
   - Backbone: `xlm-roberta-base` (278M parameters, 12 layers, 768 hidden dimensions).
   - Training: Fine-tuned for 3 epochs with AdamW ($\text{lr} = 2\times 10^{-5}$, linear warmup ratio 0.06, weight decay 0.01, mixed precision fp16, max length 128). Best checkpoint selected via validation Macro-F1.

---

## 6. Overall Benchmark Metrics

Measured on the official held-out test split ($n=19,787$):

| Metric | TF-IDF + Logistic Regression | XLM-RoBERTa (Fine-Tuned) | Absolute Gain |
|---|---|---|---|
| **Accuracy** | 82.80% | **86.12%** | **+3.32%** |
| **Macro-F1** | 0.8280 | **0.8611** | **+0.0331** |
| **Weighted-F1** | 0.8281 | **0.8612** | **+0.0331** |
| **Hate Class F1 ($F_{1, \text{hate}}$)** | 0.8230 | **0.8592** | **+0.0362** |
| **Non-Hate Class F1 ($F_{1, \text{non-hate}}$)** | 0.8330 | **0.8631** | **+0.0301** |
| **Macro Precision** | 0.8281 | **0.8611** | **+0.0330** |
| **Macro Recall** | 0.8277 | **0.8612** | **+0.0335** |
| **ROC-AUC** | 0.9091 | **0.9399** | **+0.0308** |
| **Test Confusion Matrix** | TN: 8,488 \| FP: 1,555<br>FN: 1,842 \| TP: 7,895 | TN: 8,660 \| FP: 1,390<br>FN: 1,357 \| TP: 8,380 | Fewer errors in both classes |

---

## 7. Language-Wise Performance

| Language | Test Samples ($n$) | Baseline Macro-F1 | XLM-R Macro-F1 | XLM-R Accuracy | XLM-R Hate F1 | Macro-F1 Gain |
|---|---|---|---|---|---|---|
| **English** | 1,918 | 0.7455 | **0.7836** | 79.25% | 0.8274 | **+3.81%** |
| **Hindi** | 6,723 | 0.8255 | **0.8628** | 86.30% | 0.8677 | **+3.73%** |
| **Tamil** | 5,992 | 0.8461 | **0.8788** | 87.93% | 0.8710 | **+3.27%** |
| **Malayalam** | 5,154 | 0.8325 | **0.8618** | 86.32% | 0.8478 | **+2.93%** |
| **ALL (Combined)** | 19,787 | 0.8280 | **0.8611** | 86.12% | 0.8592 | **+3.32%** |

---

## 8. Testing Category & Code-Mixed Findings
- **Indic Script Strength**: High morphological synthesis in Tamil and Malayalam is effectively handled by XLM-R's multilingual subword vocabulary, outperforming surface n-grams by nearly 3 percentage points.
- **English Sarcasm / Reclamation Challenges**: English exhibits the lowest Macro-F1 (0.7836) due to higher linguistic ambiguity (reclaimed slurs, meta-discussions of discrimination) in Twitter data.
- **Code-Mixing Status**: Code-mixed pairs (Hinglish, Tanglish, Manglish) in the 42-row manifest test subword transliteration robustness. Live per-sample scoring is pending local checkpoint restoration.

---

## 9. Confusion Matrix & Precision-Recall Analysis
- **Class Balance**: Test split contains 10,050 Non-Hate (50.79%) and 9,737 Hate (49.21%) samples.
- **XLM-R Matrix**: TN = 8,660, FP = 1,390, FN = 1,357, TP = 8,380.
- **Balance of Errors**: The model achieves nearly identical precision (0.8577) and recall (0.8606) for the hate class, confirming that the class-weighted cross-entropy loss prevented positive-class bias.

---

## 10. False-Positive and False-Negative Analysis
- **False Positive Rate (FPR)**:
  - Baseline: 15.60% (1,568 / 10,050)
  - XLM-RoBERTa: **13.83%** (1,390 / 10,050)
  - English FPR is higher (27.16%), while Tamil FPR is lowest (11.57%).
- **False Negative Rate (FNR)**:
  - Baseline: 18.83% (1,833 / 9,737)
  - XLM-RoBERTa: **13.94%** (1,357 / 9,737)
  - Transformer fine-tuning reduces false negatives by **26.0%**, significantly lowering the risk of undetected toxic comments.

---

## 11. Error Typologies & Failure Mechanisms
The error-analysis pipeline (`scripts/analyze_errors.py`) defines transparent heuristic categories:
1. **Sarcasm / Irony**: Praise adjectives juxtaposed with insults.
2. **Slang & Colloquial Pejoratives**: Obscure regional slurs with sparse token embeddings.
3. **Code-Mixing & Script Blending**: Mixed Roman and Indic scripts.
4. **Implicit Hate**: Targeted identity attacks without overt profanity.
5. **Context Dependency**: Slur reclamation and news quotations.
6. **Spelling Variations**: Character elongations (e.g. *fiiire*).
7. **Negation & Quoted Speech**: Negated toxic terms.

> [!NOTE]
> All automated error tags are tagged `review_status = "automated_flag_needs_human_verification"`. Template CSVs are generated in `results/error_analysis/` awaiting live model restoration.

---

## 12. Baseline vs. Transformer Comparison
- **TF-IDF + LR**: Fast, interpretable, but limited by vocabulary fragmentation across Indic scripts and insensitive to word order or negated polarity.
- **XLM-RoBERTa**: Excels at cross-lingual contextual representations, yielding higher F1 across all four languages, reducing both false positives and false negatives.

---

## 13. External Evaluation Findings (IndicGuard)
- **IndicGuard Broad ($n=77,677$)**: Macro-F1 = 0.5591, Accuracy = 84.88%, Hate F1 = 0.2017, FNR = **87.25%**.
- **IndicGuard Strict ($n=59,709$)**: Macro-F1 = 0.5843, Accuracy = 85.19%, Hate F1 = 0.2393, FNR = **84.34%**.
- **Domain Shift Insight**: The severe drop in Hate F1 highlights the domain mismatch between colloquial social media abuse (MACD) and translated LLM safety prompts (IndicGuard). Social classifiers cannot be deployed as out-of-domain LLM safety filters without domain adaptation.

---

## 14. System Explainability & Latency Profiles
- **Integrated Gradients**: LayerIntegratedGradients on the word embedding layer produces token attributions with completeness error validation ($\Delta_{\text{target}} \approx \sum \text{attr}$).
- **Fallback**: Occlusion sensitivity triggers automatically if Integrated Gradients completeness convergence fails.
- **Latency Benchmark (Phase 17)**:
  - Preprocessing: ~1.2 ms
  - XLM-R Forward Pass: ~38.3 ms
  - Prediction-only round trip: **39.5 ms (median)**
  - Explanation overhead: ~1.95 s (calculated only for positive hate detections)

---

## 15. Limitations & Ethical Considerations
- **Content Sensitivity**: Datasets contain offensive real-world social comments; data access is restricted to academic and safety research.
- **Reclaimed Slurs**: Automated models struggle with group-internal reclamation in colloquial English.
- **Synthetic Data Caveat**: LLM safety benchmarks translated by machine translation do not substitute for native social media hate benchmarks.

---

## 16. Teammate Checkpoint Restoration Protocol
To complete live local forward passes on the 42-row manifest:
1. Download the folder `xlm_roberta_hate/` from Google Drive:
   ```
   /content/drive/MyDrive/hate-speech/xlm_roberta_hate
   ```
   (Expected SHA-256: `fda10ac6bf81d189ab599a080cae5c159e05325a374c09797ac9245c8fb5f396`)
2. Place the folder into:
   ```
   models/xlm_roberta_hate/
   ```
3. Run the automated evaluation and error-analysis pipeline:
   ```bash
   python scripts/run_evaluation.py
   python scripts/analyze_errors.py
   ```
   This will immediately run forward inference, populate `results/predictions/test_predictions.csv`, and extract per-example false positives and false negatives into `results/error_analysis/`.

