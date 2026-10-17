"""PHASE 12 - Explainability with Integrated Gradients (IG).

Question answered: "Which input tokens received the highest attribution scores toward the predicted class?"
(NOT "which words are hate words" - see docs/explainability.md for the correct wording.)

Method
  1. Take the embedding of every token of the input  (x).
  2. Build a BASELINE of the same length: every token replaced by <pad>, except the first and last
     special tokens (<s>, </s>) which stay.
  3. Walk in `n_steps` small steps from the baseline to the real input, and at each step measure the
     gradient of P(target class) with respect to the embeddings.
  4. IG = (x - baseline) * average gradient; summing over the embedding dimension gives ONE signed
     score per token (positive = pushes toward the target class).
  5. Merge sub-word pieces back into whole words (XLM-R marks the start of a word with "▁").

Quality check: IG scores should add up to  P(target | input) - P(target | baseline).
`completeness_error` reports how far off they are (relative). Small (< ~0.05) = trustworthy;
if it is large, use more steps.
"""
from __future__ import annotations

import html

from src.data.preprocess import _has_content, clean_for_xlmr
from src.utils.config import ID2LABEL


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


def select_important_words(words: list[dict], top_k: int = 5, min_importance: float = 0.25) -> list[str]:
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
        title = f"attribution {w['score']:+.3f} | importance {w['importance']:.2f} ({w['level']})"
        parts.append(f'<span style="{style}" title="{title}">{html.escape(w["word"])}</span>')
    return '<div style="font-size:1.1em; line-height:2;">' + " ".join(parts) + "</div>"


# ----------------------------------------------------------------------------- Integrated Gradients
def explain_text(text: str, model, tokenizer, target_class: int | None = None, n_steps: int = 50,
                 internal_batch: int = 16, max_length: int = 128, top_k: int = 5,
                 min_importance: float = 0.25) -> dict:
    """Integrated Gradients for ONE text. `target_class` defaults to the predicted class."""
    import torch

    model.eval()
    dev = next(model.parameters()).device
    clean = clean_for_xlmr(text)               # same preprocessing as in training
    if not clean:
        raise ValueError("Empty text after cleaning.")
    enc = tokenizer(clean, return_tensors="pt", truncation=True, max_length=max_length).to(dev)
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]      # shapes [1, L]

    with torch.no_grad():
        probs = torch.softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits.float(), dim=-1)[0]
    pred = int(probs.argmax())
    target = pred if target_class is None else int(target_class)

    emb_layer = model.get_input_embeddings()
    with torch.no_grad():
        x = emb_layer(input_ids).detach()                                    # [1, L, D]
        base_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        base_ids[:, 0] = input_ids[:, 0]                                     # keep <s>
        base_ids[:, -1] = input_ids[:, -1]                                   # keep </s>
        base = emb_layer(base_ids).detach()
        f_base = torch.softmax(model(inputs_embeds=base, attention_mask=attention_mask).logits.float(), dim=-1)[0, target].item()
    f_x = probs[target].item()

    grads_sum = torch.zeros_like(x)
    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=dev)[1:]           # n_steps points in (0, 1]
    with torch.enable_grad():
        for chunk in alphas.split(internal_batch):
            b = chunk.shape[0]
            interp = (base + chunk.view(-1, 1, 1) * (x - base)).detach().requires_grad_(True)   # [b, L, D]
            logits = model(inputs_embeds=interp, attention_mask=attention_mask.repeat(b, 1)).logits.float()
            score = torch.softmax(logits, dim=-1)[:, target].sum()
            grad = torch.autograd.grad(score, interp)[0]                     # [b, L, D]
            grads_sum += grad.sum(dim=0, keepdim=True)
    ig = (x - base) * (grads_sum / n_steps)                                  # [1, L, D]
    token_scores = ig.sum(dim=-1)[0].detach().cpu().tolist()                 # one signed score per token

    gap = f_x - f_base
    completeness_error = abs(sum(token_scores) - gap) / max(abs(gap), 1e-8)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    special = set(tokenizer.all_special_ids)
    keep = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special]
    tokens = [all_tokens[i] for i in keep]
    scores = [token_scores[i] for i in keep]

    words = add_importance(merge_subwords(tokens, scores))
    return {
        "text": text, "clean_text": clean,
        "predicted_class": pred, "predicted_label": ID2LABEL[pred],
        "confidence": float(probs[pred]), "target_class": target, "target_label": ID2LABEL[target],
        "tokens": [{"token": t, "score": s} for t, s in zip(tokens, scores)],
        "words": words,
        "important_words": select_important_words(words, top_k=top_k, min_importance=min_importance),
        "completeness_error": float(completeness_error),
        "n_steps": n_steps,
        "truncated": bool(input_ids.shape[1] >= max_length),
    }
