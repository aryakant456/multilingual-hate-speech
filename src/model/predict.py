"""PHASE 13 - Standalone prediction pipeline (no web app needed).

    from src.model.predict import HateSpeechPredictor
    p = HateSpeechPredictor()                 # loads models/xlm_roberta_hate/ once (CPU is fine)
    p.predict("I hate those disgusting people")
    -> {"prediction": "HATE", "confidence": 0.92, "language": "English", "important_words": ["hate", "disgusting"]}

From the terminal (project root, venv active):
    python -m src.model.predict "I hate those disgusting people"

Notes
  * The text goes through the SAME minimal cleaning as in training (clean_for_xlmr).
  * `important_words` = words with the highest Integrated-Gradients attribution toward the predicted
    class. They are only computed when the prediction is HATE (nothing to highlight otherwise);
    pass explain_non_hate=True to also explain NON-HATE predictions.
  * `language` comes from the script of the text (src/utils/language.py) - see its limits.
"""
from __future__ import annotations

import json

from src.data.preprocess import clean_for_xlmr
from src.explainability.integrated_gradients import explain_text
from src.model.evaluate import load_model, predict_probs
from src.utils.config import ID2LABEL, XLMR_DIR
from src.utils.language import detect_language


class HateSpeechPredictor:
    def __init__(self, model_dir=XLMR_DIR, device: str | None = None, max_length: int = 128,
                 threshold: float = 0.5, n_steps: int = 50):
        self.tokenizer, self.model, self.device = load_model(model_dir, device)
        self.max_length, self.threshold, self.n_steps = max_length, threshold, n_steps

    def predict(self, text: str, explain: bool = True, top_k: int = 5, details: bool = False,
                explain_non_hate: bool = False) -> dict:
        clean = clean_for_xlmr(text)
        if not clean:
            raise ValueError("The text is empty after cleaning.")
        p_non, p_hate = (float(x) for x in predict_probs(self.model, self.tokenizer, [clean],
                                                         max_length=self.max_length, show_progress=False)[0])
        pred = int(p_hate >= self.threshold)
        lang = detect_language(clean)
        important, expl = [], None
        if explain and (pred == 1 or explain_non_hate):
            expl = explain_text(clean, self.model, self.tokenizer, target_class=pred, n_steps=self.n_steps,
                                max_length=self.max_length, top_k=top_k)
            important = expl["important_words"]
        out = {"prediction": ID2LABEL[pred], "confidence": round(p_hate if pred == 1 else p_non, 4),
               "language": lang["language"], "important_words": important}
        if details:
            out["details"] = {"p_hate": round(p_hate, 4), "p_non_hate": round(p_non, 4), "script": lang["script"],
                              "script_share": lang["share"], "threshold": self.threshold,
                              "explanation": None if expl is None else {
                                  "completeness_error": round(expl["completeness_error"], 4),
                                  "n_steps": expl["n_steps"], "truncated": expl["truncated"],
                                  "words": [{"word": w["word"], "score": round(w["score"], 4),
                                             "importance": round(w["importance"], 3)} for w in expl["words"]]}}
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Predict hate / non-hate for one text")
    ap.add_argument("text")
    ap.add_argument("--details", action="store_true")
    a = ap.parse_args()
    print(json.dumps(HateSpeechPredictor().predict(a.text, details=a.details), ensure_ascii=False, indent=2))
