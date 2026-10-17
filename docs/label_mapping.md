# Label mapping (Phase 3)

Unified label used everywhere in this project: **0 = NON-HATE / NON-ABUSIVE, 1 = HATE / ABUSIVE.**
Every conversion below was checked against the real downloaded files by `python -m src.data.combine_datasets`
(it stops with an error if it finds a raw label it does not expect). Evidence: `results/metrics/label_mapping.json`.

| Source | Raw label in the file | Unified label | Notes |
|---|---|---|---|
| MACD | `0` | **1 (Hate/Abusive)** | **Inverted on purpose.** In MACD, 0 means abusive. |
| MACD | `1` | **0 (Non-Hate)** | Verified: sum of raw `0` over `data/raw/macd/dataset/` = 74,550 = the paper's "abusive samples"; raw `1` = 77,872 = "non-abusive samples". |
| HateXplain | `hatespeech` | 1 | Majority vote of the 3 annotators. |
| HateXplain | `offensive` | 1 | Deliberate decision: the task is "hate OR abusive". The 3-class label stays in `label_raw`. |
| HateXplain | `normal` | 0 | |
| HateXplain | `undecided` (3 annotators all differ) | dropped | As in the original paper. |
| IndicGuard | `violated_categories` contains `Hate/Identity Hate` | 1 | IndicGuard is an LLM-safety dataset; this category is the closest match to hate. Used only as an external test set. |
| IndicGuard | `prompt_label = safe` | 0 | |
| IndicGuard | unsafe for other reasons only (e.g. Criminal Planning) | dropped | Not "normal" text, so they must not count as non-hate. |

Other facts that were verified and are used by the code
* MACD: only `data/raw/macd/dataset/` is used (the other folders are re-splits of the same comments). Columns `text`, `label`. Languages: Hindi, Tamil, Malayalam, Telugu, Kannada. The file name spells Malayalam as `malyalam`.
* Unified columns: `text, label, language, source, split, label_raw` (+ `category, tag` for IndicGuard).
