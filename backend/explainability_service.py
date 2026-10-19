"""Explanation service (Phase 14): important words for a HATE prediction.

Order of methods
  1. Integrated Gradients (best of four baselines, adaptive steps) - used when its own checks pass;
  2. occlusion (leave-one-word-out) - automatic fallback when the Integrated-Gradients checks fail.
The returned dict says which one was used, so the UI can be honest about it.
"""
from __future__ import annotations

from src.explainability.integrated_gradients import explain_occlusion, explain_text


class ExplainabilityService:
    def __init__(self, model, tokenizer, max_length: int = 128, n_steps: int = 50, max_steps: int = 100,
                 fallback_occlusion: bool = True):
        self.model, self.tokenizer = model, tokenizer
        self.max_length, self.n_steps, self.max_steps = max_length, n_steps, max_steps
        self.fallback_occlusion = fallback_occlusion

    def explain(self, clean_text: str, target_class: int, top_k: int = 5) -> dict:
        ig = explain_text(clean_text, self.model, self.tokenizer, target_class=target_class, n_steps=self.n_steps,
                          max_steps=self.max_steps, max_length=self.max_length, top_k=top_k)
        used, method = ig, "integrated_gradients"
        if not ig["reliable"] and self.fallback_occlusion:
            used = explain_occlusion(clean_text, self.model, self.tokenizer, target_class=target_class,
                                     max_length=self.max_length, top_k=top_k)
            method = "occlusion"
        return {
            "method": method,
            "important_words": used["important_words"],
            "words": [{"word": w["word"], "score": round(float(w["score"]), 4),
                       "importance": round(float(w["importance"]), 3), "level": w["level"]} for w in used["words"]],
            "reliable": bool(used["reliable"]),
            "baseline": ig.get("baseline"),
            "n_steps": ig.get("n_steps"),
            "warning": ig.get("warning") if method == "integrated_gradients" else
                       ("Integrated Gradients failed its reliability check, so the occlusion method was used. "
                        + (ig.get("warning") or "")).strip(),
            "truncated": bool(ig.get("truncated", False)),
        }
