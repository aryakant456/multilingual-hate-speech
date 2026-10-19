"""Model loading and prediction service (Phase 14) - with the stage timings needed for Phase 17.

Pipeline inside `analyze` (this IS the application flow of Phase 16):
    clean text -> language detection -> [XLM-R tokenizer -> fine-tuned XLM-R] -> prediction + confidence
               -> (HATE only) Integrated Gradients -> important words -> response

Timings (milliseconds, measured with time.perf_counter):
    preprocessing_ms   cleaning + language detection + token count
    inference_ms       tokenizer + XLM-R forward pass + softmax
    explainability_ms  Integrated Gradients (+ occlusion fallback); 0 if no explanation was computed
    total_ms           whole `analyze` call (server side; the HTTP round trip is measured by the client)

The model is loaded ONCE (at server start) and a lock makes requests run one at a time, so timings are not
distorted by two requests fighting for the same CPU.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from backend.explainability_service import ExplainabilityService
from src.data.preprocess import clean_for_xlmr
from src.model.evaluate import load_model, predict_probs
from src.utils.config import ID2LABEL, XLMR_DIR
from src.utils.language import detect_language

logger = logging.getLogger("hate_api")

MODEL_DIR = Path(os.getenv("HATE_MODEL_DIR", str(XLMR_DIR)))
MAX_LENGTH = int(os.getenv("HATE_MAX_LENGTH", "128"))
IG_STEPS = int(os.getenv("HATE_IG_STEPS", "50"))
IG_MAX_STEPS = int(os.getenv("HATE_IG_MAX_STEPS", "100"))


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 2)


class ModelService:
    def __init__(self, model_dir=None, max_length: int = MAX_LENGTH, threshold: float = 0.5,
                 explain_non_hate: bool = False):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.max_length, self.threshold, self.explain_non_hate = max_length, threshold, explain_non_hate
        self.model = self.tokenizer = self.explainer = self.device = None
        self.load_seconds: float | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        t0 = time.perf_counter()
        try:
            self.tokenizer, self.model, self.device = load_model(self.model_dir)
            self.explainer = ExplainabilityService(self.model, self.tokenizer, max_length=self.max_length,
                                                   n_steps=IG_STEPS, max_steps=IG_MAX_STEPS)
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        self.error = None
        self.load_seconds = round(time.perf_counter() - t0, 2)

    def info(self) -> dict:
        return {"status": "ok" if self.ready else "model_not_loaded", "model_loaded": self.ready,
                "device": str(self.device) if self.device is not None else None,
                "model_dir": str(self.model_dir), "load_seconds": self.load_seconds, "error": self.error}

    # ------------------------------------------------------------------ main entry point
    def analyze(self, text: str, explain: bool = True, details: bool = False, top_k: int = 5) -> dict:
        if not self.ready:
            raise RuntimeError("Model not loaded")
        with self._lock:
            t0 = time.perf_counter()

            # 1. preprocessing: same cleaning as in training, language detection, token count
            clean = clean_for_xlmr(text)
            if not clean:
                raise ValueError("The text is empty after cleaning (only invisible characters or markup).")
            lang = detect_language(clean)
            n_tokens = len(self.tokenizer(clean, truncation=False)["input_ids"])
            t1 = time.perf_counter()

            # 2. tokenizer + fine-tuned XLM-R -> prediction and confidence
            p_non, p_hate = (float(x) for x in predict_probs(self.model, self.tokenizer, [clean],
                                                             max_length=self.max_length, show_progress=False)[0])
            pred = int(p_hate >= self.threshold)
            t2 = time.perf_counter()

            # 3. explainability (only for HATE predictions unless configured otherwise)
            expl = None
            if explain and (pred == 1 or self.explain_non_hate):
                expl = self.explainer.explain(clean, target_class=pred, top_k=top_k)
            t3 = time.perf_counter()

        timings = {"preprocessing_ms": _ms(t1 - t0), "inference_ms": _ms(t2 - t1),
                   "explainability_ms": _ms(t3 - t2) if expl is not None else 0.0,    # exactly 0 when skipped
                   "total_ms": _ms(t3 - t0)}
        logger.info("predict lang=%s pred=%s chars=%d tokens=%d total=%.0fms",     # the text itself is never logged
                    lang["language"], ID2LABEL[pred], len(text), n_tokens, timings["total_ms"])

        out = {"prediction": ID2LABEL[pred], "confidence": round(p_hate if pred == 1 else p_non, 4),
               "language": lang["language"], "important_words": expl["important_words"] if expl else []}
        if details:
            warnings = []
            if n_tokens > self.max_length:
                warnings.append(f"The text has {n_tokens} tokens; only the first {self.max_length} were analysed.")
            if lang["language"] == "Unknown":
                warnings.append("The language could not be determined from the script.")
            if expl and expl.get("warning"):
                warnings.append(expl["warning"])
            out.update({
                "clean_text": clean,
                "words": expl["words"] if expl else [],
                "explanation": None if expl is None else {k: expl[k] for k in ("method", "reliable", "baseline", "n_steps", "warning")},
                "warnings": warnings,
                "timings_ms": timings,
            })
        return out
