"""PHASE 12 - Explainability: Integrated Gradients (IG) with a baseline check, plus an occlusion fallback.

Question answered: "Which input tokens received the highest attribution scores toward the predicted class?"
(NOT "which words are hate words" - see docs/explainability.md for the correct wording.)

Integrated Gradients, step by step
  1. embed every token of the input (x);
  2. choose a BASELINE (a "no information" input of the same length);
  3. walk from the baseline to x in `n_steps` small steps (midpoint rule) and average the gradient of
     P(target class) with respect to the embeddings;
  4. IG = (x - baseline) * average gradient; summing over the embedding dimension gives one signed score per token;
  5. merge sub-word pieces back into words (XLM-R marks word starts with "▁").

Why a baseline CHECK is needed
  IG explains the CHANGE in P(target) between baseline and input:   sum(scores) = P(x) - P(baseline).
  If the baseline itself already gets (almost) the same probability, there is nothing to explain and the scores
  are meaningless. So we (a) try several baselines (pad / mask / unk / zero) and pick the one with the LOWEST
  P(target), (b) report the gap, (c) report how well the scores add up (absolute error, in probability units),
  and (d) mark the explanation `reliable` only if the gap is large enough and the error small.

Fallback: `explain_occlusion` removes one word at a time and measures the drop in P(target). It needs no
baseline, so it is used when IG is not reliable.
"""
from __future__ import annotations

import html

from src.data.preprocess import _has_content, clean_for_xlmr
from src.model.evaluate import predict_probs
from src.utils.config import ID2LABEL

BASELINES = ("pad", "mask", "unk", "zero")


# ----------------------------------------------------------------------------- pure helpers
def merge_subwords(tokens: list[str], scores: list[float]) -> list[dict]:
    """Group sentencepiece tokens into words. A token starting with '▁' begins a new word;
    other tokens are glued to the previous word. Word score = sum of its token scores."""
    words: list[dict] = []
    cur = None
    for tok, sc in zip(tokens, scores):
        starts = tok.startswith("▁")
        piece = tok.lstrip("▁") if starts else tok
        if starts or cur is None:
            if cur is not None:
                words.append(cur)
            cur = {"word": piece, "score": float(sc), "tokens": [tok]}
        else:
            cur["word"] += piece
            cur["score"] += float(sc)
            cur["tokens"].append(tok)
    if cur is not None:
        words.append(cur)
    return [w for w in words if w["word"] != ""]


def importance_level(importance: float) -> str:
    """Strong / medium / low contribution (the three tiers used in the UI)."""
    if importance >= 0.66:
        return "strong"
    if importance >= 0.33:
        return "medium"
    if importance >= 0.10:
        return "low"
    return "none"


def add_importance(words: list[dict]) -> list[dict]:
    """importance = positive score / largest positive score  (0..1); negative scores -> 0."""
    pos_max = max([w["score"] for w in words if w["score"] > 0], default=0.0)
    for w in words:
        w["importance"] = (max(w["score"], 0.0) / pos_max) if pos_max > 0 else 0.0
        w["level"] = importance_level(w["importance"])
    return words


def attribution_bar(importance: float) -> str:
    """Text bar: strong ████████, medium ████, low ██."""
    return {"strong": "████████", "medium": "████", "low": "██", "none": ""}[importance_level(importance)]


_EDGE_PUNCT = ".,!?;:\"'()[]{}<>-_*/\\|\u2026\u0964\u0965\u201c\u201d\u2018\u2019"


def select_important_words(words: list[dict], top_k: int = 5, min_importance: float = 0.10) -> list[str]:
    """Top words by importance. Punctuation stuck to the edge of a word is removed ("people!" -> "people")."""
    cand = [w for w in words if w["score"] > 0 and w["importance"] >= min_importance and _has_content(w["word"])]
    cand.sort(key=lambda w: w["importance"], reverse=True)
    out: list[str] = []
    for w in cand:
        shown = w["word"].strip(_EDGE_PUNCT) or w["word"]
        if shown not in out:
            out.append(shown)
        if len(out) == top_k:
            break
    return out


def render_html(words: list[dict], rgb: str = "220, 38, 38") -> str:
    """HTML where every word is highlighted with an intensity proportional to its importance.
    Use in a notebook with  IPython.display.HTML(render_html(words))  - the web UI can reuse it."""
    parts = []
    for w in words:
        alpha = 0.0 if w["importance"] < 0.10 else 0.15 + 0.75 * w["importance"]
        style = f"background: rgba({rgb}, {alpha:.2f}); padding: 1px 3px; border-radius: 3px;" if alpha else ""
        title = f"score {w['score']:+.3f} | importance {w['importance']:.2f} ({w['level']})"
        parts.append(f'<span style="{style}" title="{title}">{html.escape(w["word"])}</span>')
    return '<div style="font-size:1.1em; line-height:2;">' + " ".join(parts) + "</div>"


def choose_baseline(baseline_probs: dict, requested: str = "auto") -> str:
    """'auto' = the baseline with the LOWEST probability of the target class (largest probability change to explain)."""
    if requested != "auto":
        if requested not in baseline_probs:
            raise ValueError(f"unknown baseline '{requested}'; choose from {sorted(baseline_probs)} or 'auto'")
        return requested
    return min(baseline_probs, key=lambda k: baseline_probs[k])


def assess_reliability(gap: float, abs_error: float, tolerance: float = 0.05, min_gap: float = 0.20,
                       baseline: str = "", p_base: float = 0.0, p_input: float = 0.0, label: str = "") -> tuple[bool, str | None]:
    """Reliable only if (1) the baseline leaves enough probability change to explain and (2) the scores add up."""
    if gap < min_gap:
        return False, (f"Baseline '{baseline}' already gives P({label}) = {p_base:.2f} (input: {p_input:.2f}); the change to "
                       f"explain is only {gap:.2f}, so the Integrated-Gradients scores are NOT reliable for this text.")
    if abs_error > tolerance:
        return False, (f"The scores miss the probability change by {abs_error:.3f} (limit {tolerance}); "
                       "increase n_steps / max_steps.")
    return True, None


# ----------------------------------------------------------------------------- torch part: Integrated Gradients
def _baseline_embeds(name: str, input_ids, x, tokenizer, emb_layer):
    """Baseline embeddings [1, L, D]. First (<s>) and last (</s>) token are always kept."""
    import torch
    if name == "zero":
        base = torch.zeros_like(x)
        base[:, 0], base[:, -1] = x[:, 0], x[:, -1]
        return base
    tid = {"pad": tokenizer.pad_token_id, "mask": tokenizer.mask_token_id, "unk": tokenizer.unk_token_id}[name]
    ids = torch.full_like(input_ids, tid)
    ids[:, 0], ids[:, -1] = input_ids[:, 0], input_ids[:, -1]
    with torch.no_grad():
        return emb_layer(ids).detach()


def _integrated_gradients(model, x, base, attention_mask, target: int, n_steps: int, internal_batch: int):
    """IG tensor [1, L, D] with the midpoint rule (more accurate than the plain Riemann sum for the same cost)."""
    import torch
    grads_sum = torch.zeros_like(x)
    alphas = (torch.arange(n_steps, dtype=torch.float32, device=x.device) + 0.5) / n_steps
    with torch.enable_grad():
        for chunk in alphas.split(internal_batch):
            b = chunk.shape[0]
            interp = (base + chunk.view(-1, 1, 1) * (x - base)).detach().requires_grad_(True)     # [b, L, D]
            logits = model(inputs_embeds=interp, attention_mask=attention_mask.repeat(b, 1)).logits.float()
            score = torch.softmax(logits, dim=-1)[:, target].sum()
            grad = torch.autograd.grad(score, interp)[0]
            grads_sum += grad.sum(dim=0, keepdim=True)
    return (x - base) * (grads_sum / n_steps)


def explain_text(text: str, model, tokenizer, target_class: int | None = None, n_steps: int = 50,
                 baseline: str = "auto", adaptive: bool = True, max_steps: int = 200, tolerance: float = 0.05,
                 min_gap: float = 0.20, internal_batch: int = 16, max_length: int = 128, top_k: int = 5,
                 min_importance: float = 0.10) -> dict:
    """Integrated Gradients for ONE text. `target_class` defaults to the predicted class.
    adaptive=True doubles n_steps (up to max_steps) until the scores add up (absolute error <= tolerance)."""
    import torch

    model.eval()
    dev = next(model.parameters()).device
    clean = clean_for_xlmr(text)               # same preprocessing as in training
    if not clean:
        raise ValueError("Empty text after cleaning.")
    enc = tokenizer(clean, return_tensors="pt", truncation=True, max_length=max_length).to(dev)
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]      # [1, L]

    with torch.no_grad():
        probs = torch.softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits.float(), dim=-1)[0]
    pred = int(probs.argmax())
    target = pred if target_class is None else int(target_class)
    f_x = probs[target].item()

    emb_layer = model.get_input_embeddings()
    with torch.no_grad():
        x = emb_layer(input_ids).detach()                                    # [1, L, D]
        base_emb, base_probs = {}, {}
        for name in BASELINES:                                               # diagnostic: P(target) for every baseline
            base_emb[name] = _baseline_embeds(name, input_ids, x, tokenizer, emb_layer)
            logits = model(inputs_embeds=base_emb[name], attention_mask=attention_mask).logits.float()
            base_probs[name] = torch.softmax(logits, dim=-1)[0, target].item()
    chosen = choose_baseline(base_probs, baseline)
    f_b = base_probs[chosen]
    gap = f_x - f_b

    steps = n_steps
    while True:
        ig = _integrated_gradients(model, x, base_emb[chosen], attention_mask, target, steps, internal_batch)
        token_scores = ig.sum(dim=-1)[0].detach().cpu().tolist()             # one signed score per token
        abs_error = abs(sum(token_scores) - gap)
        if not adaptive or abs_error <= tolerance or steps >= max_steps:
            break
        steps = min(steps * 2, max_steps)
    reliable, warning = assess_reliability(gap, abs_error, tolerance, min_gap, chosen, f_b, f_x, ID2LABEL[target])

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    special = set(tokenizer.all_special_ids)
    keep = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special]
    words = add_importance(merge_subwords([all_tokens[i] for i in keep], [token_scores[i] for i in keep]))
    return {
        "method": "integrated_gradients",
        "text": text, "clean_text": clean,
        "predicted_class": pred, "predicted_label": ID2LABEL[pred],
        "confidence": float(probs[pred]), "target_class": target, "target_label": ID2LABEL[target],
        "tokens": [{"token": all_tokens[i], "score": token_scores[i]} for i in keep],
        "words": words,
        "important_words": select_important_words(words, top_k=top_k, min_importance=min_importance),
        "baseline": chosen, "baseline_probabilities": {k: round(v, 4) for k, v in base_probs.items()},
        "p_input": f_x, "p_baseline": f_b, "gap": gap,
        "completeness_abs_error": float(abs_error),
        "completeness_error": float(abs_error / max(abs(gap), 1e-8)),      # relative (unstable when gap is tiny)
        "reliable": reliable, "warning": warning,
        "n_steps": steps, "truncated": bool(input_ids.shape[1] >= max_length),
    }


# ----------------------------------------------------------------------------- fallback: occlusion
def explain_occlusion(text: str, model, tokenizer, target_class: int | None = None, max_length: int = 128,
                      top_k: int = 5, min_importance: float = 0.10) -> dict:
    """Leave-one-word-out: score(word) = P(target | full text) - P(target | text without that word).
    Needs no baseline, so it works when Integrated Gradients is not reliable. One forward pass per word."""
    clean = clean_for_xlmr(text)
    if not clean:
        raise ValueError("Empty text after cleaning.")
    toks = clean.split(" ")
    variants = [clean] + [" ".join(toks[:i] + toks[i + 1:]) for i in range(len(toks))]
    probs = predict_probs(model, tokenizer, variants, max_length=max_length, show_progress=False)
    pred = int(probs[0].argmax())
    target = pred if target_class is None else int(target_class)
    words = [{"word": w, "score": float(probs[0, target] - probs[i + 1, target]), "tokens": [w]}
             for i, w in enumerate(toks)]
    words = add_importance(words)
    return {
        "method": "occlusion", "text": text, "clean_text": clean,
        "predicted_class": pred, "predicted_label": ID2LABEL[pred], "confidence": float(probs[0, pred]),
        "target_class": target, "target_label": ID2LABEL[target],
        "words": words, "important_words": select_important_words(words, top_k=top_k, min_importance=min_importance),
        "reliable": True, "warning": None,
    }
