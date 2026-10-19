# Application flow (Phase 16)

```
User
 |  types text, clicks ANALYZE                          frontend/src/App.jsx
 v
React (Vite, port 5173)
 |  POST /api/predict  ->  Vite proxy  ->  POST /predict   frontend/src/api.js, vite.config.js
 v
FastAPI (port 8000)                                     backend/main.py
 |  validates the request (1-2000 characters)           backend/schemas.py
 v
Preprocessing + Language detection                      backend/model_service.py -> src/data/preprocess.py (clean_for_xlmr)
 |                                                                               -> src/utils/language.py  (detect_language)
 v
XLM-R tokenizer -> fine-tuned XLM-R                     src/model/evaluate.py (predict_probs), models/xlm_roberta_hate/
 |
 +--> Prediction (HATE / NON-HATE)
 +--> Confidence (probability of the predicted class)
 |
 v   (only when the prediction is HATE and "explain" is true)
Integrated Gradients (best of 4 baselines, adaptive steps)   backend/explainability_service.py
 |   -> occlusion fallback when the reliability check fails      -> src/explainability/integrated_gradients.py
 v
Important words + word-level highlights
 v
JSON response  { prediction, confidence, language, important_words [, words, explanation, warnings, timings_ms] }
 v
React shows: Language, Prediction, Confidence, Important words, highlighted sentence, response time
```

## API contract
`POST /predict`

Request: `{"text": "I hate those disgusting people", "explain": true, "details": false}`

Response (default, exactly the project specification):
```json
{"prediction": "HATE", "confidence": 0.92, "language": "English", "important_words": ["hate", "disgusting"]}
```
With `"details": true` the response also contains `clean_text`, `words` (word, score, importance, level),
`explanation` (method, reliable, baseline, n_steps, warning), `warnings` and `timings_ms`
(preprocessing_ms, inference_ms, explainability_ms, total_ms). The React app always asks for details.

Errors: 422 = invalid request (empty, blank or longer than 2000 characters), 400 = nothing left after cleaning,
503 = model not loaded (see `GET /health`).

## Design decisions
* The model is loaded once when the server starts; requests run one at a time (a lock) so timings are clean.
* `explain=false` is a fast path that returns only the prediction.
* The user's text is never written to the server log - only language, label, length and timings.
* Important words are computed only for HATE predictions.
* Language comes from the script of the text (limits: Devanagari -> "Hindi" also covers Marathi; Latin -> "English" also covers romanised Hindi/Tamil).
