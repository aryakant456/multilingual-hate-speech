"""End-to-end check of the running API (Phases 14 and 16).

1. Start the backend in one terminal:   uvicorn backend.main:app --host 127.0.0.1 --port 8000
2. In another terminal:                  python scripts/check_api.py            (optional: --url http://127.0.0.1:8000)

Every line is PASS or FAIL. It checks the exact output format required by the project, the explanation
fields, the timings, several languages and the error handling. Uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

results: list[tuple[bool, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((bool(ok), name))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -  {detail}" if detail else ""))


def call(base: str, path: str, payload: dict | None = None):
    """Returns (http status, parsed json or None)."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, method="POST" if payload is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return e.code, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    base = ap.parse_args().url.rstrip("/")

    try:
        status, health = call(base, "/health")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Cannot reach {base} ({exc}). Start the backend first: uvicorn backend.main:app --port 8000")
    check(status == 200 and health and health.get("model_loaded") is True, "GET /health: model loaded",
          f"device={health.get('device')}, load={health.get('load_seconds')} s" if health else "")
    if not (health and health.get("model_loaded")):
        raise SystemExit(f"Model not loaded: {health.get('error') if health else '?'}")

    # --- the exact output format of the project specification -------------------------------------
    s, r = call(base, "/predict", {"text": "I hate those disgusting people"})
    check(s == 200 and r is not None and list(r) == ["prediction", "confidence", "language", "important_words"],
          "POST /predict returns exactly: prediction, confidence, language, important_words", str(r))
    if r:
        check(r["prediction"] in ("HATE", "NON-HATE") and 0 <= r["confidence"] <= 1, "prediction label and confidence range")
        check(r["language"] == "English", "language detected: English", r["language"])
        check(isinstance(r["important_words"], list), "important_words is a list", str(r["important_words"]))

    # --- details: highlights, method, timings -------------------------------------------------------
    t0 = time.perf_counter()
    s, d = call(base, "/predict", {"text": "I hate those disgusting people", "details": True})
    wall = (time.perf_counter() - t0) * 1000
    check(s == 200 and d is not None and {"clean_text", "words", "explanation", "warnings", "timings_ms"} <= set(d), "details=true adds words, explanation, warnings, timings_ms")
    if d and d.get("timings_ms"):
        t = d["timings_ms"]
        check(set(t) == {"preprocessing_ms", "inference_ms", "explainability_ms", "total_ms"}, "four timing stages present", str(t))
        check(t["total_ms"] <= wall + 5, "server total_ms is not larger than the measured round trip", f"server {t['total_ms']:.0f} ms vs round trip {wall:.0f} ms")
    if d and d["prediction"] == "HATE":
        check(len(d["words"]) > 0 and all({"word", "score", "importance", "level"} <= set(w) for w in d["words"]), "HATE: every word has score, importance, level")
        check(d["explanation"] and d["explanation"]["method"] in ("integrated_gradients", "occlusion"), "HATE: explanation method reported", str(d["explanation"]))

    # --- friendly text in all four languages -> NON-HATE, no highlights ------------------------------------------
    for text, lang in [("Have a wonderful day my friend", "English"), ("आप सब लोग बहुत अच्छे हैं", "Hindi"),
                       ("இந்த வீடியோ மிகவும் அருமை", "Tamil"), ("ഈ വീഡിയോ വളരെ നന്നായിട്ടുണ്ട്", "Malayalam")]:
        s, r = call(base, "/predict", {"text": text})
        check(s == 200 and r and r["language"] == lang, f"language {lang} detected", str(r and r["language"]))
        check(r and r["prediction"] == "NON-HATE" and r["important_words"] == [], f"{lang} friendly text: NON-HATE, no important words",
              str(r and (r["prediction"], round(r["confidence"], 3))))

    # --- fast path -------------------------------------------------------------------------------------------------------
    s, r = call(base, "/predict", {"text": "I hate those disgusting people", "explain": False, "details": True})
    check(s == 200 and r and r["important_words"] == [] and r["timings_ms"]["explainability_ms"] == 0, "explain=false skips the explanation (fast path)")

    # --- error handling ------------------------------------------------------------------------------------------------------
    for payload, name in [({"text": ""}, "empty text"), ({"text": "   "}, "blank text"), ({"text": "a" * 2001}, "text longer than 2000 characters"), ({}, "missing text field")]:
        s, _ = call(base, "/predict", payload)
        check(s == 422, f"rejected with HTTP 422: {name}", f"got {s}")
    s, _ = call(base, "/predict", {"text": "\u200b\u200b"})
    check(s == 400, "invisible-only text rejected with HTTP 400", f"got {s}")

    failed = [n for ok, n in results if not ok]
    print(f"\nSUMMARY: {len(results) - len(failed)} PASS, {len(failed)} FAIL")
    print("=> Backend OK." if not failed else "=> Fix the FAIL lines above (send me the output if unsure).")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
