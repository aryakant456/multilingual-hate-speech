"""Tiny dataset + batching helpers shared by training and evaluation (needs torch only when batching)."""
from __future__ import annotations


class TextDataset:
    """Holds texts (and optionally labels). Works with torch's DataLoader (needs only __len__/__getitem__)."""

    def __init__(self, texts, labels=None):
        self.texts = [("" if t is None else str(t)) for t in texts]
        self.labels = None if labels is None else [int(x) for x in labels]
        if self.labels is not None and len(self.labels) != len(self.texts):
            raise ValueError("texts and labels must have the same length")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        item = {"text": self.texts[i]}
        if self.labels is not None:
            item["label"] = self.labels[i]
        return item


def make_collate_fn(tokenizer, max_length: int = 128):
    """Returns a function that turns a list of items into one padded batch:
    text -> XLM-R tokenizer -> input_ids + attention_mask (+ labels).
    Padding is DYNAMIC: each batch is padded only to its own longest text (max `max_length` tokens)."""

    def collate(batch: list[dict]):
        import torch
        enc = tokenizer([b["text"] for b in batch], padding=True, truncation=True,
                        max_length=max_length, pad_to_multiple_of=8, return_tensors="pt")
        if "label" in batch[0]:
            enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return enc

    return collate
