"""PhoBERT-base text encoder for the modular VQA pipeline (A1/A2).

Wraps ``vinai/phobert-base`` to produce per-token hidden states ``(B, T, D)``
that the co-attention fusion module consumes. Frozen by default; the last
``unfreeze_last_n`` transformer layers can be unfrozen for fine-tuning.

Set ``KAGGLE_NO_INTERNET=1`` to force ``local_files_only=True`` on both the
model and tokenizer downloads — required when running on a Kaggle accelerator
without internet access.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn


DEFAULT_MODEL_ID: str = "vinai/phobert-base"
DEFAULT_MAX_LENGTH: int = 64


def _local_files_only() -> bool:
    return os.environ.get("KAGGLE_NO_INTERNET") == "1"


class PhoBERTTextEncoder(nn.Module):
    """PhoBERT-base wrapper returning per-token hidden states.

    Args:
        model_id: HuggingFace repo id. Defaults to ``vinai/phobert-base``.
        freeze: If ``True`` (default), all parameters are frozen.
        unfreeze_last_n: Number of trailing transformer layers to unfreeze
            when ``freeze=True``. Ignored when ``freeze=False``.

    Output:
        ``forward(input_ids, attention_mask)`` returns a tensor of shape
        ``(B, T, hidden_size)`` (last hidden state). The pooled ``[CLS]``
        embedding is available as ``out[:, 0]`` if the caller wants it.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        freeze: bool = True,
        unfreeze_last_n: int = 0,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model_id = model_id
        self.model = AutoModel.from_pretrained(
            model_id, local_files_only=_local_files_only()
        )
        self.hidden_size: int = int(self.model.config.hidden_size)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            if unfreeze_last_n > 0:
                self._unfreeze_last_n_layers(unfreeze_last_n)

    def _unfreeze_last_n_layers(self, n: int) -> None:
        layers = self.model.encoder.layer
        n = min(n, len(layers))
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state


def load_phobert_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
) -> Any:
    """Load the PhoBERT tokenizer used to feed :class:`PhoBERTTextEncoder`."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id, local_files_only=_local_files_only(), use_fast=False
    )


def encode_questions(
    tokenizer: Any,
    questions: list[str],
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of Vietnamese questions for the encoder.

    Returns a dict with ``input_ids`` and ``attention_mask`` tensors padded to
    ``max_length`` (or the longest sequence in the batch, whichever is smaller).
    """
    return tokenizer(
        questions,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
