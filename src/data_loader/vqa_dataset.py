"""Torch ``Dataset`` for FoodLensVN modular pipeline (A1/A2).

Reads a processed split JSON (output of ``scripts/build_dataset.py``) and
returns one dict per row, ready for the collate function in
:mod:`src.data_loader.collate`.

Each row in the input JSON has been canonicalized by the build pipeline; this
class only:
  * resolves the image path (``images_root / row["image"]``) and applies the
    backbone-aware transform from :func:`build_image_transform`,
  * encodes the answer via :class:`AnswerTokenizer` to a fixed-length id list,
  * passes the raw question string through (collate-time PhoBERT tokenization),
  * forwards the metadata fields (``type``, ``difficulty``, ``dish``, ``id``)
    so eval can compute per-type / per-difficulty / per-dish breakdowns.

The dataset never loads the answer vocab itself — pass an
:class:`AnswerTokenizer` constructed from
``data/processed/answer_vocab.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data_loader.answer_tokenizer import AnswerTokenizer


class VQADataset(Dataset):
    def __init__(
        self,
        rows_path: str | Path,
        images_root: str | Path,
        image_transform: Callable[[Image.Image], torch.Tensor],
        answer_tokenizer: AnswerTokenizer,
    ) -> None:
        rows_path = Path(rows_path)
        with rows_path.open("r", encoding="utf-8") as f:
            self.rows: list[dict] = json.load(f)
        if not isinstance(self.rows, list):
            raise ValueError(
                f"{rows_path} must contain a JSON list of rows, "
                f"got {type(self.rows).__name__}"
            )

        self.images_root = Path(images_root)
        self.image_transform = image_transform
        self.answer_tokenizer = answer_tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = self.images_root / row["image"]
        with Image.open(image_path) as im:
            pixel_values = self.image_transform(im.convert("RGB"))

        answer_ids = self.answer_tokenizer.encode(row["answer"])

        return {
            "pixel_values": pixel_values,
            "question": row["question"],
            "answer_ids": torch.as_tensor(answer_ids, dtype=torch.long),
            "id": row["id"],
            "type": row["type"],
            "difficulty": row["difficulty"],
            "dish": row["dish"],
        }
