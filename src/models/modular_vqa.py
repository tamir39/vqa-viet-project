"""End-to-end modular VQA model assembly for A1/A2.

Wires four pre-built components into one ``nn.Module``:

    image_encoder  -> img_feats     (B, P,  D_img)
    text_encoder   -> txt_feats     (B, T,  D_txt)
    fusion         -> (fused_seq, fused_pooled)
    decoder        -> logits / generated ids

The decoder consumes either the pooled vector (LSTM, A1) or the fused
sequence (Transformer, A2). Routing is selected by ``decoder_input`` at
construction time:

    decoder_input="pooled"  ->  feeds fused_pooled to the decoder  (A1)
    decoder_input="seq"     ->  feeds fused_seq    to the decoder  (A2)

The model is intentionally agnostic to the concrete component classes so
it can later be reused for ablations (different encoders, fusion variants,
or a third decoder family) without modifying this file.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


DecoderInput = Literal["pooled", "seq"]


class ModularVQA(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        fusion: nn.Module,
        decoder: nn.Module,
        decoder_input: DecoderInput = "pooled",
    ) -> None:
        super().__init__()
        if decoder_input not in ("pooled", "seq"):
            raise ValueError(
                f"decoder_input must be 'pooled' or 'seq', got {decoder_input!r}"
            )
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion = fusion
        self.decoder = decoder
        self.decoder_input = decoder_input

    def _encode(
        self,
        pixel_values: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        img_feats = self.image_encoder(pixel_values)
        txt_feats = self.text_encoder(question_input_ids, question_attention_mask)
        fused_seq, fused_pooled = self.fusion(
            img_feats, txt_feats, question_attention_mask
        )
        if self.decoder_input == "seq":
            if fused_seq is None:
                raise ValueError(
                    "decoder_input='seq' requires a fusion module that produces "
                    "fused_seq (use CoAttentionFusion)."
                )
            return fused_seq, fused_pooled
        return None, fused_pooled

    def forward(
        self,
        pixel_values: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        fused_seq, fused_pooled = self._encode(
            pixel_values, question_input_ids, question_attention_mask
        )
        ctx = fused_pooled if self.decoder_input == "pooled" else fused_seq
        return self.decoder(ctx, target_ids)

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        max_len: int | None = None,
    ) -> torch.Tensor:
        fused_seq, fused_pooled = self._encode(
            pixel_values, question_input_ids, question_attention_mask
        )
        ctx = fused_pooled if self.decoder_input == "pooled" else fused_seq
        return self.decoder.generate(ctx, max_len=max_len)
