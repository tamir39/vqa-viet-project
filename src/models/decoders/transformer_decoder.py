"""Transformer decoder for the modular VQA pipeline (A2).

Consumes the fused token sequence from co-attention fusion as cross-attention
memory and decodes auto-regressively over the :class:`AnswerTokenizer`
vocabulary (PAD/BOS/EOS/UNK at ids 0–3, ``MAX_LEN=12``).

Forward signature
    forward(fused_seq: (B, M, D_fused), target_ids: (B, T)) -> logits (B, T, V)
Generation
    generate(fused_seq) -> ids (B, <=MAX_LEN), starting with BOS, padded after EOS.

The fusion module that pairs with this decoder is ``CoAttentionFusion`` — it
returns the concatenated refined image+text sequence as ``fused_seq``. The
ablation fusions (elementwise/concat) emit ``fused_seq=None`` and are
incompatible with this decoder by design.
"""

from __future__ import annotations

import torch
from torch import nn

from src.data_loader.answer_tokenizer import AnswerTokenizer


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_fused: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = AnswerTokenizer.MAX_LEN,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.embed = nn.Embedding(
            vocab_size, d_model, padding_idx=AnswerTokenizer.PAD_ID
        )
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.mem_proj = (
            nn.Linear(d_fused, d_model) if d_fused != d_model else nn.Identity()
        )

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _embed_with_pos(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        return self.embed(ids) + self.pos_embed(positions)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

    def _require_seq(self, fused_seq: torch.Tensor | None) -> torch.Tensor:
        if fused_seq is None:
            raise ValueError(
                "TransformerDecoder requires fused_seq from the fusion module; "
                "use CoAttentionFusion (elementwise/concat ablations are pooled-only)."
            )
        return fused_seq

    def forward(
        self,
        fused_seq: torch.Tensor | None,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.mem_proj(self._require_seq(fused_seq))
        tgt_emb = self._embed_with_pos(target_ids)
        tgt_mask = self._causal_mask(target_ids.size(1), target_ids.device)
        tgt_key_padding_mask = target_ids == AnswerTokenizer.PAD_ID
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.out_proj(out)

    @torch.no_grad()
    def generate(
        self,
        fused_seq: torch.Tensor | None,
        max_len: int | None = None,
    ) -> torch.Tensor:
        memory = self.mem_proj(self._require_seq(fused_seq))
        max_len = max_len or self.max_len
        B = memory.size(0)
        device = memory.device

        ids = torch.full(
            (B, 1), AnswerTokenizer.BOS_ID, dtype=torch.long, device=device
        )
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_emb = self._embed_with_pos(ids)
            tgt_mask = self._causal_mask(ids.size(1), device)
            out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            logits = self.out_proj(out[:, -1])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            next_tok = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_tok, AnswerTokenizer.PAD_ID),
                next_tok,
            )
            ids = torch.cat([ids, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(-1) == AnswerTokenizer.EOS_ID)
            if finished.all():
                break

        return ids
