"""PHASE 17 - Real-time performance: measure the inference time of the full pipeline.

Run from the project root (virtual environment active):
    python scripts/benchmark_latency.py                              # in-process, no web server needed
    python scripts/benchmark_latency.py --http http://127.0.0.1:8000 # through the running FastAPI server
    python scripts/benchmark_latency.py --test-sample 20             # also use 20 real test comments

For every input the pipeline is run `--repeats` times (after `--warmup` discarded calls) and the MEDIAN of each
stage is reported:
    preprocessing   cleaning + language detection + token count
    inference       tokenizer + XLM-R forward pass
    explainability  Integrated Gradients (+ occlusion fallback) - only computed for HATE predictions
    total           whole server-side call.   (--http mode also records the client-side round trip)

Writes: results/metrics/latency_benchmark.csv / .json / .md  and  results/figures/latency_breakdown.png
Nothing here is estimated - every number comes from time.perf_counter() on YOUR machine.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUILTIN_INPUTS = [  # harmless / mild examples written for this benchmark
    "I hate those disgusting people",
    "You are a stupid idiot and everyone hates you",
    "Have a wonderful day my friend",
    "Thank you so much for sharing this video, it was really helpful and I learned a lot from it",
    "आप सब लोग बहुत अच्छे हैं",
    "இந்த வீடியோ மிகவும் அருமை",
    "ഈ വീഡിയോ വളരെ നന്നായിട്ടുണ്ട്",
    "I hate those disgusting people. " * 12,     # long text: hits the 128-token limit
]
STAGES = ["preprocessing_ms", "inference_ms", "explainability_ms", "total_ms"]


# ------------------------------------------------------------------------------------ inputs
def build_inputs(extra_file: str | None = None, test_sample: int = 0) -> list[dict]:
    items = [{"source": "built-in", "text": t, "true_label": None} for t in BUILTIN_INPUTS]
    if extra_file:
        for line in Path(extra_file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append({"source": "file", "text": line.strip(), "true_label": None})
    if test_sample > 0:
        import pandas as pd
        from src.data.io_utils import read_processed_csv
        from src.utils.config import TEST_FILE
        df = read_processed_csv(TEST_FILE)
        half = test_sample // 2
        picked = pd.concat([df[df.label == 1].sample(half, random_state=42), df[df.label == 0].sample(test_sample - half, random_state=42)])
        for _, r in picked.iterrows():
            items.append({"source": f"test set ({r['language']})", "text": str(r["text"]), "true_label": int(r["label"])})
    return items


# ------------------------------------------------------------------------------------ measuring
def http_analyze(base_url: str, text: str, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(base_url.rstrip("/") + "/predict", method="POST",
                                 data=json.dumps({"text": text, "explain": True, "details": True}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def measure(analyze, inputs: list[dict], repeats: int) -> list[dict]:
    """analyze(text) -> response dict with 'timings_ms'. Returns one summary row per input."""
    rows = []
    for idx, item in enumerate(inputs, start=1):
        samples, out = [], None
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = analyze(item["text"])
            client_ms = (time.perf_counter() - t0) * 1000.0
            samples.append({**out["timings_ms"], "client_ms": client_ms})
        med = lambda key: float(np.median([s[key] for s in samples]))
        explained = out.get("explanation") is not None
        rows.append({
            "id": idx, "source": item["source"], "chars": len(item["text"]), "language": out["language"],
            "prediction": out["prediction"], "confidence": out["confidence"], "true_label": item["true_label"],
            "explained": explained, "method": out["explanation"]["method"] if explained else "-",
            "preprocessing_ms": med("preprocessing_ms"), "inference_ms": med("inference_ms"),
            "explainability_ms": med("explainability_ms"), "total_ms": med("total_ms"),
            "total_min_ms": float(min(s["total_ms"] for s in samples)), "total_max_ms": float(max(s["total_ms"] for s in samples)),
            "client_ms": med("client_ms"), "repeats": repeats,
            "text_preview": item["text"][:60] if item["source"] in ("built-in", "file") else "",
            "_samples": samples,
        })
    return rows


def stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median_ms": round(float(np.median(a)), 1), "mean_ms": round(float(a.mean()), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1), "min_ms": round(float(a.min()), 1), "max_ms": round(float(a.max()), 1)}


def summarise(rows: list[dict]) -> dict:
    all_s = [s for r in rows for s in r["_samples"]]
    hate = [s for r in rows if r["explained"] for s in r["_samples"]]
    non = [s for r in rows if not r["explained"] for s in r["_samples"]]
    out = {
        "prediction_only_ms (preprocessing + inference, all inputs)": stats([s["preprocessing_ms"] + s["inference_ms"] for s in all_s]),
        "total_ms, no explanation (NON-HATE inputs)": stats([s["total_ms"] for s in non]),
        "total_ms, with explanation (HATE inputs)": stats([s["total_ms"] for s in hate]),
        "explainability_ms (HATE inputs)": stats([s["explainability_ms"] for s in hate]),
    }
    if hate:
        tot = sum(s["total_ms"] for s in hate)
        out["explainability_share_of_total_for_hate_inputs_%"] = round(100 * sum(s["explainability_ms"] for s in hate) / tot, 1) if tot else None
    return out


def environment(device_hint: str | None = None) -> dict:
    env = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
           "python": platform.python_version()}
    try:
        import torch
        import transformers
        env.update({"torch": torch.__version__, "transformers": transformers.__version__,
                    "torch_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available()})
    except ImportError:
        pass
    if device_hint:
        env["device"] = device_hint
    return env


# ------------------------------------------------------------------------------------ outputs
def to_markdown(rows: list[dict]) -> str:
    head = "| # | Input | Prediction | Confidence | Preprocessing | Inference | Explainability | Total |\n|---|---|---|---|---|---|---|---|\n"
    lines = []
    for r in rows:
        label = r["text_preview"].replace("|", "/").replace("\n", " ") + ("..." if r["chars"] > 60 and r["text_preview"] else "")
        label = label or f"{r['source']}, {r['chars']} chars"
        lines.append(f"| {r['id']} | {label} | {r['prediction']} | {100 * r['confidence']:.1f}% | {r['preprocessing_ms']:.1f} ms | "
                     f"{r['inference_ms']:.1f} ms | {r['explainability_ms']:.1f} ms | {r['total_ms']:.1f} ms |")
    return head + "\n".join(lines) + "\n"


def plot_breakdown(rows: list[dict], path) -> None:
    from matplotlib.figure import Figure
    fig = Figure(figsize=(max(7, 0.9 * len(rows) + 2), 4.5))
    ax = fig.subplots()
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for key, label, color in [("preprocessing_ms", "preprocessing", "#9ca3af"), ("inference_ms", "model inference", "#2563eb"),
                              ("explainability_ms", "explainability", "#dc2626")]:
        vals = np.array([r[key] for r in rows])
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_xticks(x, labels=[str(r["id"]) for r in rows])
    ax.set_xlabel("input #")
    ax.set_ylabel("median time (ms)")
    ax.set_title("Response time per input and pipeline stage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write_outputs(rows, summary, meta, out_dir: Path, fig_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cols = ["id", "source", "chars", "language", "prediction", "confidence", "true_label", "explained", "method",
            "preprocessing_ms", "inference_ms", "explainability_ms", "total_ms", "total_min_ms", "total_max_ms", "client_ms", "repeats"]
    with open(out_dir / "latency_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k in cols})
    with open(out_dir / "latency_benchmark.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "summary": summary}, f, indent=2, ensure_ascii=False)
    with open(out_dir / "latency_benchmark.md", "w", encoding="utf-8") as f:
        f.write(to_markdown(rows))
    plot_breakdown(rows, fig_dir / "latency_breakdown.png")


# ------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the response time of the hate-speech pipeline")
    ap.add_argument("--repeats", type=int, default=5, help="measured runs per input (median is reported)")
    ap.add_argument("--warmup", type=int, default=3, help="discarded runs before measuring")
    ap.add_argument("--http", default=None, help="base URL of a running server, e.g. http://127.0.0.1:8000")
    ap.add_argument("--inputs", default=None, help="text file with one extra input per line")
    ap.add_argument("--test-sample", type=int, default=0, help="also use N random comments from data/processed/test.csv")
    a = ap.parse_args()

    from src.utils.config import FIGURES_DIR, METRICS_DIR
    inputs = build_inputs(a.inputs, a.test_sample)
    meta = {"mode": "http" if a.http else "in-process", "repeats": a.repeats, "warmup": a.warmup, "n_inputs": len(inputs)}

    if a.http:
        analyze = lambda text: http_analyze(a.http, text)
        health = json.loads(urllib.request.urlopen(a.http.rstrip("/") + "/health", timeout=30).read().decode("utf-8"))
        if not health.get("model_loaded"):
            raise SystemExit(f"The server is up but the model is not loaded: {health.get('error')}")
        meta["server"] = health
        env = environment(health.get("device"))
        print(f"Measuring through the HTTP server at {a.http}")
    else:
        from backend.model_service import ModelService
        svc = ModelService()
        t0 = time.perf_counter()
        svc.load()
        meta["model_load_seconds"] = round(time.perf_counter() - t0, 2)
        analyze = lambda text: svc.analyze(text, explain=True, details=True)
        env = environment(str(svc.device))
        print(f"Model loaded in {meta['model_load_seconds']} s on {svc.device}")

    t0 = time.perf_counter()
    analyze(inputs[0]["text"])
    meta["first_request_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    print(f"First request (cold): {meta['first_request_ms']} ms; warming up ({a.warmup} runs) ...")
    for _ in range(a.warmup):
        analyze(inputs[0]["text"])

    rows = measure(analyze, inputs, a.repeats)
    summary = summarise(rows)
    meta["environment"] = env
    write_outputs(rows, summary, meta, METRICS_DIR, FIGURES_DIR)

    print("\n" + to_markdown(rows))
    print("Summary (milliseconds):")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nEnvironment: {env}")
    print("Saved: results/metrics/latency_benchmark.csv|json|md and results/figures/latency_breakdown.png")


if __name__ == "__main__":
    main()
