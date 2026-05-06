"""Collate function for the modular VQA DataLoader.

The dataset returns one dict per row with raw question strings (so PhoBERT
batched padding can handle them efficiently). :class:`VQACollate` bundles a
batch into:

    {
        "pixel_values":          (B, C, H, W) float,
        "question_input_ids":    (B, Lq)      long,
        "question_attention_mask": (B, Lq)    long,
        "answer_ids":            (B, MAX_LEN) long,
        "meta": [
            {"id": ..., "type": ..., "difficulty": ..., "dish": ...},
            ...
        ],
    }

The collate is a class (not a closure) so it survives ``forkserver`` /
``spawn`` pickling — required under Python 3.14 where DataLoader workers
default to ``forkserver`` on Linux. :func:`make_collate_fn` is kept as a thin
factory for backwards compatibility with existing callers.
"""

from __future__ import annotations

from typing import Any

import torch


class VQACollate:
    """Picklable collate callable for the modular VQA DataLoader.

    Args:
        phobert_tokenizer: HuggingFace tokenizer for PhoBERT
            (e.g., from :func:`src.models.encoders.text_encoder.load_phobert_tokenizer`).
        max_question_length: Truncation/padding length for tokenized questions.
    """

    def __init__(self, phobert_tokenizer: Any, max_question_length: int = 64) -> None:
        self.phobert_tokenizer = phobert_tokenizer
        self.max_question_length = max_question_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        answer_ids = torch.stack([b["answer_ids"] for b in batch], dim=0)

        questions = [b["question"] for b in batch]
        tok = self.phobert_tokenizer(
            questions,
            padding="max_length",
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )

        meta = [
            {
                "id": b["id"],
                "type": b["type"],
                "difficulty": b["difficulty"],
                "dish": b["dish"],
            }
            for b in batch
        ]

        return {
            "pixel_values": pixel_values,
            "question_input_ids": tok["input_ids"],
            "question_attention_mask": tok["attention_mask"],
            "answer_ids": answer_ids,
            "meta": meta,
        }


def make_collate_fn(
    phobert_tokenizer: Any, max_question_length: int = 64
) -> VQACollate:
    """Construct a :class:`VQACollate` (kept for API stability)."""
    return VQACollate(phobert_tokenizer, max_question_length)
