"""Collate function for the modular VQA DataLoader.

The dataset returns one dict per row with raw question strings (so PhoBERT
batched padding can handle them efficiently). :func:`make_collate_fn` returns
a closure that bundles a batch into:

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

The PhoBERT tokenizer is captured by the closure so the collate fn stays
``DataLoader``-pickle-friendly under ``num_workers > 0``.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def make_collate_fn(
    phobert_tokenizer: Any,
    max_question_length: int = 64,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build a collate fn that batch-tokenizes questions with PhoBERT.

    Args:
        phobert_tokenizer: HuggingFace tokenizer for PhoBERT
            (e.g., from :func:`src.models.encoders.text_encoder.load_phobert_tokenizer`).
        max_question_length: Truncation/padding length for tokenized questions.
    """

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        answer_ids = torch.stack([b["answer_ids"] for b in batch], dim=0)

        questions = [b["question"] for b in batch]
        tok = phobert_tokenizer(
            questions,
            padding="max_length",
            truncation=True,
            max_length=max_question_length,
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

    return collate
