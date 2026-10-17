"""PHASE 9 - Fine-tune XLM-RoBERTa-base for hate / non-hate classification.

Meant to run on a GPU (Google Colab) - see notebooks/04_xlm_training.ipynb.

    XLM-R tokenizer -> input_ids + attention_mask -> XLM-RoBERTa-base -> classification head (2 classes)

Design (kept deliberately conservative - no hyper-parameter sweep):
  * max_length 128, learning rate 2e-5, batch size 32, 3 epochs, linear warm-up + decay
  * class-weighted cross-entropy (weights computed from the training labels; ~1.0 here because
    the data is almost balanced)
  * after EVERY epoch: validation loss, macro-F1, precision, recall are computed
  * the epoch with the best validation MACRO-F1 is the one that is saved
  * saved: model, tokenizer, config, label mapping, training config, training history

Only the TRAIN and VALIDATION files are read here. The test set, the zero-shot set and
IndicGuard are not touched during training or model selection.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.data.io_utils import read_processed_csv
from src.model.data import TextDataset, make_collate_fn
from src.utils.config import (ID2LABEL, LABEL2ID, METRICS_DIR, RANDOM_SEED,
                              TRAIN_FILE, VAL_FILE, XLMR_DIR, XLMR_MODEL_NAME)
from src.utils.metrics import compute_metrics


@dataclass
class TrainConfig:
    model_name: str = XLMR_MODEL_NAME
    max_length: int = 128
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 32            # lower to 16 if the GPU runs out of memory
    eval_batch_size: int = 128
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    use_class_weights: bool = True
    fp16: bool = True               # mixed precision (only used on a CUDA GPU)
    patience: int = 2               # stop early if val macro-F1 does not improve for this many epochs
    seed: int = RANDOM_SEED
    output_dir: str = str(XLMR_DIR)
    train_file: str = str(TRAIN_FILE)
    val_file: str = str(VAL_FILE)
    max_train_samples: int | None = None   # smoke test: train on a small random subset
    max_val_samples: int | None = None


# ----------------------------------------------------------------------------- helpers
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def balanced_class_weights(labels) -> list[float]:
    """w_c = n_samples / (n_classes * n_c)   (same formula as sklearn 'balanced')."""
    y = np.asarray(labels, dtype=int)
    n = len(y)
    return [float(n / (2 * max((y == c).sum(), 1))) for c in (0, 1)]


def epoch_metrics(y_true, probs: np.ndarray, loss: float) -> dict:
    """Validation metrics for one epoch. probs has shape [n, 2] (softmax probabilities)."""
    pred = probs.argmax(axis=1)
    m = compute_metrics(y_true, pred, probs[:, 1])
    return {"val_loss": float(loss), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            "precision_hate": m["precision_hate"], "recall_hate": m["recall_hate"], "f1_hate": m["f1_hate"],
            "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"], "roc_auc": m.get("roc_auc")}


def run_eval(model, loader, loss_fn, device, use_amp: bool):
    """One pass over a validation loader. Returns (mean loss, probabilities [n,2], labels)."""
    import torch
    model.eval()
    total, n = 0.0, 0
    all_probs, all_labels = [], []
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(**inputs).logits
            logits = logits.float()
            total += loss_fn(logits, labels).item() * labels.size(0)
            n += labels.size(0)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return total / max(n, 1), np.concatenate(all_probs), np.concatenate(all_labels)


def _save_best(model, tokenizer, out_dir: Path, cfg: TrainConfig, metrics: dict, epoch: int) -> None:
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(out_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID, "positive_class": "HATE"}, f, indent=2)
    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), "best_epoch": epoch, "val_metrics": metrics}, f, indent=2)


# ----------------------------------------------------------------------------- training
def train(cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig()
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.fp16 and device.type == "cuda"
    if device.type == "cpu":
        print("WARNING: no GPU found - training on CPU is extremely slow. In Colab choose "
              "Runtime > Change runtime type > T4 GPU.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))

    # ---- data -----------------------------------------------------------------
    train_df = read_processed_csv(cfg.train_file)
    val_df = read_processed_csv(cfg.val_file)
    for d in (train_df, val_df):
        d["text"] = d["text"].fillna("").astype(str)
    if cfg.max_train_samples:
        train_df = train_df.sample(n=min(cfg.max_train_samples, len(train_df)), random_state=cfg.seed).reset_index(drop=True)
    if cfg.max_val_samples:
        val_df = val_df.sample(n=min(cfg.max_val_samples, len(val_df)), random_state=cfg.seed).reset_index(drop=True)
    print(f"train {len(train_df):,} rows | val {len(val_df):,} rows")

    # ---- model + tokenizer ------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID).to(device)

    collate = make_collate_fn(tokenizer, cfg.max_length)
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    train_loader = DataLoader(TextDataset(train_df["text"], train_df["label"]), batch_size=cfg.batch_size,
                              shuffle=True, collate_fn=collate, generator=gen)
    val_loader = DataLoader(TextDataset(val_df["text"], val_df["label"]), batch_size=cfg.eval_batch_size,
                            shuffle=False, collate_fn=collate)

    # ---- optimiser, schedule, loss ------------------------------------------------
    no_decay = ("bias", "LayerNorm.weight")
    params = list(model.named_parameters())
    groups = [
        {"params": [p for n, p in params if not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=cfg.learning_rate)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)

    weights = balanced_class_weights(train_df["label"]) if cfg.use_class_weights else [1.0, 1.0]
    print("class weights [non-hate, hate]:", [round(w, 4) for w in weights])
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):                      # older torch versions
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    # ---- epochs --------------------------------------------------------------------
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_f1, best_epoch, bad_epochs = [], -1.0, 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for batch in bar:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(**inputs).logits
            loss = loss_fn(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            bar.set_postfix(loss=round(running / seen, 4))

        val_loss, probs, y_val = run_eval(model, val_loader, loss_fn, device, use_amp)
        m = epoch_metrics(y_val, probs, val_loss)
        entry = {"epoch": epoch, "train_loss": running / max(seen, 1), **m,
                 "minutes_elapsed": round((time.time() - t0) / 60, 1)}
        history.append(entry)
        print(f"epoch {epoch}: train loss {entry['train_loss']:.4f} | val loss {m['val_loss']:.4f} | "
              f"macro-F1 {m['macro_f1']:.4f} | precision(hate) {m['precision_hate']:.4f} | "
              f"recall(hate) {m['recall_hate']:.4f}")

        if m["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = m["macro_f1"], epoch, 0
            _save_best(model, tokenizer, out_dir, cfg, m, epoch)
            print(f"   -> new best (macro-F1 {best_f1:.4f}); model saved to {out_dir}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"   early stop: no improvement for {cfg.patience} epochs")
                break

    result = {"history": history, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
              "output_dir": str(out_dir), "train_minutes": round((time.time() - t0) / 60, 1)}
    for target in (out_dir / "training_history.json", METRICS_DIR / "xlmr_training_history.json"):
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(f"\nDone in {result['train_minutes']} min. Best epoch {best_epoch}, val macro-F1 {best_f1:.4f}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fine-tune XLM-R (run on a GPU)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--output-dir", default=str(XLMR_DIR))
    ap.add_argument("--smoke", action="store_true", help="2000 training rows, 1 epoch")
    a = ap.parse_args()
    train(TrainConfig(epochs=1 if a.smoke else a.epochs, batch_size=a.batch_size, max_length=a.max_length,
                      learning_rate=a.lr, output_dir=a.output_dir,
                      max_train_samples=2000 if a.smoke else None, max_val_samples=1000 if a.smoke else None))
