# Explainability (Phase 12)

Method: **Integrated Gradients** on the XLM-R input embeddings (`src/explainability/integrated_gradients.py`).

* Baseline: same length as the input; four candidates are tried (every token replaced by `<pad>`, `<mask>`, `<unk>`, or a zero vector;
  the first `<s>` and last `</s>` stay) and the one with the LOWEST P(predicted class) is used ("auto").
  Reason: IG explains the change between baseline and input; if the baseline already gets the same probability there is nothing to explain.
* `n_steps = 50` interpolation steps (midpoint rule), doubled up to 200 until the scores add up; gradient of P(predicted class) with respect to the embeddings.
* One signed score per token (sum over the embedding dimension); sub-word pieces are merged into words
  (XLM-R marks word starts with `▁`); word importance = positive score / largest positive score.
* Checks per explanation: `gap` = P(class | input) - P(class | baseline) must be at least 0.20, and the sum of the scores must match
  the gap within 0.05 (absolute, in probability units). Otherwise the explanation is marked NOT reliable.
* Fallback: occlusion (leave-one-word-out). Score of a word = drop in P(predicted class) when it is removed. It needs no baseline.
  `predict()` uses it automatically when Integrated Gradients is not reliable (`details=True` shows which method was used).
* Importance tiers for the UI: strong (>= 0.66), medium (>= 0.33), low (>= 0.10).

## Wording for the report
Do NOT write: *"The model identified these words as hate words."*
Write: *"The explainability module identifies the tokens with the highest attribution scores toward the predicted hate class."*

Why: attribution scores describe how the model arrived at its output. They are not a ground-truth statement about which words
are hateful, and they depend on the baseline and the number of steps.

## Known limits
* Attributions are computed for the predicted class; for NON-HATE predictions `predict()` returns no important words by default.
* Context matters: a word can score high in one sentence and low in another.
* Language shown by `predict()` comes from the script of the text (Devanagari -> "Hindi" also covers Marathi; Latin -> "English" also covers romanised Hindi/Tamil).
