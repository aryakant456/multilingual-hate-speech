# Phase 21 — Final Model Comparison Report

## 1. Executive Summary & Objective
Phase 21 provides the definitive comparative synthesis of the Multilingual Hate and Abusive Speech Detection project. It consolidates experimental metrics across model architectures, in-domain evaluation languages (English, Hindi, Tamil, Malayalam), and external safety benchmarks (IndicGuard).

In strict adherence to research integrity standards:
- All reported metrics are verified against primary experimental outputs in `results/metrics/`.
- No model weights were retrained, and no predictions were fabricated.
- The 42-row dedicated multilingual manifest (`data/evaluation/multilingual_test_manifest.csv`) is preserved and recognized as un-evaluated locally pending checkpoint restoration.
- Per-example error analysis remains designated as **Pending** because the local error CSVs contain schema templates awaiting model inference.

---

## 2. In-Domain Benchmark Comparison (Official Test Set, $n=19,787$)

The official held-out test split combines **MACD** (Hindi, Tamil, Malayalam) and **HateXplain** (English), providing a rigorously balanced test set:

| Model Architecture | Evaluated Split | Test Samples ($n$) | Accuracy | Macro-F1 | Weighted-F1 | Hate F1 | Non-Hate F1 | Macro Precision | Macro Recall | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | Official Held-Out Test | 19,787 | 82.80% | 0.8280 | 0.8281 | 0.8230 | 0.8330 | 0.8281 | 0.8277 | 0.9091 |
| **XLM-RoBERTa (Fine-Tuned)** | Official Held-Out Test | 19,787 | **86.12%** | **0.8611** | **0.8612** | **0.8592** | **0.8631** | **0.8611** | **0.8612** | **0.9399** |
| **Absolute Gain ($\Delta$)** | — | — | **+3.32%** | **+0.0331** | **+0.0331** | **+0.0362** | **+0.0301** | **+0.0330** | **+0.0335** | **+0.0308** |

*Source files*: `results/metrics/baseline_metrics.json`, `results/metrics/xlmr_test_metrics.json`, and `results/metrics/final_model_comparison.csv`.

### Key Analytical Takeaways:
1. **Transformer Representation Advantage**: Fine-tuning XLM-RoBERTa yields consistent improvements across all aggregate metrics (+3.32% Accuracy, +3.31% Macro-F1).
2. **Hate Class Disambiguation**: The largest single gain is observed in the positive hate class ($F_{1, \text{hate}}$ increases from 0.8230 to 0.8592, a +3.62% gain). Contextual self-attention enables the model to distinguish benign occurrences of sensitive identity terms from malicious targeted abuse.
3. **High Discriminative Capacity**: An ROC-AUC of 0.9399 confirms high threshold-independent separation between hate and non-hate distributions.

---

## 3. Language-Wise Performance Analysis

Performance breakdown across the four primary languages on the official held-out test split:

| Language | Test Samples ($n$) | Baseline Accuracy | Baseline Macro-F1 | XLM-R Accuracy | XLM-R Macro-F1 | XLM-R Hate F1 | Macro-F1 Gain | XLM-R FPR (%) | XLM-R FNR (%) |
|---|---|---|---|---|---|---|---|---|---|
| **English** | 1,918 | 75.70% | 0.7455 | 79.25% | **0.7836** | 0.8274 | **+0.0381 (+3.81%)** | 27.16% | 16.39% |
| **Hindi** | 6,723 | 82.58% | 0.8255 | 86.30% | **0.8628** | 0.8677 | **+0.0373 (+3.73%)** | 13.87% | 13.54% |
| **Tamil** | 5,992 | 84.65% | 0.8461 | 87.93% | **0.8788** | 0.8710 | **+0.0327 (+3.27%)** | 11.57% | 12.63% |
| **Malayalam** | 5,154 | 83.41% | 0.8325 | 86.32% | **0.8618** | 0.8478 | **+0.0293 (+2.93%)** | 12.68% | 14.90% |
| **ALL (Combined)** | 19,787 | 82.80% | 0.8280 | 86.12% | **0.8611** | 0.8592 | **+0.0332 (+3.32%)** | 13.83% | 13.94% |

*Source files*: `results/metrics/model_comparison.csv` and `results/metrics/final_language_comparison.csv`.

### Language-Specific Strengths & Weaknesses:
- **Tamil (Highest Performance, 0.8788 Macro-F1)**:
  - Demonstrates the lowest FPR (11.57%) and lowest FNR (12.63%). Native script agglutinative morphology is effectively captured by SentencePiece subword tokenization.
- **Hindi (0.8628 Macro-F1)** and **Malayalam (0.8618 Macro-F1)**:
  - Both show balanced, robust performance with near-identical error rates (~13%–14%).
- **English (Challenging Nuance, 0.7836 Macro-F1)**:
  - While English achieves the largest absolute Macro-F1 gain (+3.81%) over the baseline, its overall Macro-F1 remains lower than the Indic languages.
  - HateXplain contains conversational Twitter/Reddit discourse with extensive sarcasm, political polemics, and reclaimed slurs. This results in a higher False Positive Rate (27.16%), where benign posts containing profanity are flagged as hate.

---

## 4. In-Domain Test vs. Out-of-Domain IndicGuard Evaluation

To test model generalization beyond social media comments, the locked XLM-RoBERTa model was evaluated on **IndicGuard** (`l3cube-pune/IndicGuard`), an LLM safety evaluation corpus of translated prompts:

| Evaluation Protocol | Dataset | Samples ($n$) | Accuracy | Macro-F1 | Hate F1 | Non-Hate F1 | ROC-AUC | FPR (%) | FNR (%) |
|---|---|---|---|---|---|---|---|---|---|
| **In-Domain Internal** | MACD + HateXplain | 19,787 | 86.12% | **0.8611** | 0.8592 | 0.8631 | 0.9399 | 13.83% | 13.94% |
| **External Broad** | IndicGuard (All Prompts) | 77,677 | 84.88% | **0.5591** | 0.2017 | 0.9165 | 0.7516 | 2.40% | **87.25%** |
| **External Strict** | IndicGuard (No Jailbreaks) | 59,709 | 85.19% | **0.5843** | 0.2393 | 0.9189 | 0.7937 | 2.65% | **84.34%** |

*Source files*: `results/metrics/xlmr_internal_vs_external.csv` and `results/metrics/indicguard_metrics.json`.

### Critical Methodological Caveats:
> [!WARNING]
> Direct comparison between in-domain test scores (0.8611 Macro-F1) and IndicGuard scores (0.5591 Macro-F1) is **not** an indicator of model degradation, but evidence of profound **domain shift**:
> 1. **Data Genre Shift**: Training was conducted on organic social media abusive comments (MACD/HateXplain). IndicGuard consists of synthetic, machine-translated LLM prompts designed to test safety guardrails.
> 2. **Extreme Class Imbalance**: Hate constitutes ~50% of the training and internal test data, but only ~15% of IndicGuard.
> 3. **High False Negative Rate (87.25%)**: The model rarely falsely accuses safe prompts (FPR is only 2.40%), but misses adversarial persona/jailbreak prompts that do not contain colloquial abusive language.
> 4. **Translation Artifacts**: IndicGuard prompts were machine-translated across 11 Indic languages from English AEGIS benchmarks, introducing unnatural phrasings absent in real user comments.

---

## 5. Status of the Dedicated Evaluation Manifest (42 Samples)

- **Manifest Location**: `data/evaluation/multilingual_test_manifest.csv`
- **Integrity**: 42 rows $\times$ 7 columns, 0 missing texts, 0 duplicates.
- **Composition**:
  - English: 4 Normal, 4 Hateful, 4 Ambiguous (needs adjudication)
  - Hindi: 4 Normal, 4 Hateful
  - Tamil: 4 Normal, 4 Hateful
  - Malayalam: 4 Normal, 4 Hateful
  - Code-Mixed: 2 Hinglish, 2 Tanglish, 2 Manglish
- **Status**: **Un-evaluated locally**. In accordance with instructions, no synthetic model predictions were generated. Live forward evaluation on this manifest will execute as soon as model weights are placed in `models/xlm_roberta_hate/`.

---

## 6. Error Analysis Status

- **Status**: **Pending Per-Sample Inference**.
- The error-analysis pipeline script (`scripts/analyze_errors.py`) is fully implemented, verified, and unit-tested.
- The output files (`results/error_analysis/false_positives.csv` and `results/error_analysis/false_negatives.csv`) currently hold valid schema headers (0 data rows). They are clearly designated as **templates awaiting live prediction inputs**, not completed empirical extractions.
- Aggregate error rates (FPR and FNR) are documented from verified benchmark metric JSONs.

---

## 7. Model Checkpoint Location & Teammate Restoration Checklist

The fine-tuned model checkpoint must be restored from Google Drive:
- **Location on Google Drive**: `/content/drive/MyDrive/hate-speech/xlm_roberta_hate`
- **Model Checkpoint SHA-256 Fingerprint**: `fda10ac6bf81d189ab599a080cae5c159e05325a374c09797ac9245c8fb5f396`
- **Target Local Workspace Folder**: `models/xlm_roberta_hate/`

### Required Files to Copy:
1. `config.json`
2. `model.safetensors` (or `pytorch_model.bin`) (~1.1 GB)
3. `tokenizer.json`
4. `tokenizer_config.json`
5. `sentencepiece.bpe.model`
6. `special_tokens_map.json`
7. `training_config.json`
8. `label_mapping.json`

Once copied into `models/xlm_roberta_hate/`, execute:
```bash
python scripts/run_evaluation.py
python scripts/analyze_errors.py
```
This will immediately run live forward inference on the 42-row manifest, populate `results/predictions/test_predictions.csv`, and extract per-example false positives and false negatives.

