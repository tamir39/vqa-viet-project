"""Vision-language fusion modules for the modular VQA pipeline (A1/A2).

Three variants share a common interface:

    forward(img_feats, txt_feats, txt_mask=None) -> (fused_seq, fused_pooled)

* ``CoAttentionFusion`` (primary) — projects both modalities into a shared
  ``d_fused`` space, runs cross-attention in both directions
  (image→text, text→image), then concatenates the refined sequences and
  mean-pools (mask-aware on the text side) to a single ``(B, d_fused)``
  vector. ``fused_seq`` (the concatenated refined sequence) is exposed for
  the Transformer decoder; ``fused_pooled`` feeds the LSTM decoder.

* ``ElementwiseFusion`` (ablation) — pools both sides, projects to
  ``d_fused``, and multiplies. ``fused_seq`` is ``None``.

* ``ConcatFusion`` (ablation) — pools both sides, concatenates, projects
  back to ``d_fused``. ``fused_seq`` is ``None``.

Use :func:`build_fusion` as the factory; pick the variant via the
``fusion_type`` config field.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


FusionType = Literal["co_attention", "elementwise", "concat"]


def _masked_mean(seq: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool over the sequence dim, respecting an optional padding mask.

    ``seq``: ``(B, T, D)``. ``mask``: ``(B, T)`` with 1 for valid tokens,
    0 for padding. Returns ``(B, D)``.
    """
    if mask is None:
        return seq.mean(dim=1)
    mask = mask.to(seq.dtype).unsqueeze(-1)
    summed = (seq * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


class _CrossAttentionBlock(nn.Module):
    """Single-direction multi-head cross-attention with residual + LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )
        return self.norm(query + attended)


class CoAttentionFusion(nn.Module):
    """Bidirectional co-attention fusion (primary).

    Args:
        d_img: image-feature channel dim ``D_img`` from the image encoder.
        d_txt: text-feature channel dim ``D_txt`` from the text encoder.
        d_fused: shared fusion dim.
        n_heads: number of attention heads (``d_fused`` must be divisible).
        dropout: dropout inside attention.
    """

    def __init__(
        self,
        d_img: int,
        d_txt: int,
        d_fused: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_fused % n_heads != 0:
            raise ValueError(
                f"d_fused={d_fused} must be divisible by n_heads={n_heads}"
            )
        self.d_fused = d_fused
        self.proj_img = nn.Linear(d_img, d_fused)
        self.proj_txt = nn.Linear(d_txt, d_fused)
        self.img_attends_txt = _CrossAttentionBlock(d_fused, n_heads, dropout)
        self.txt_attends_img = _CrossAttentionBlock(d_fused, n_heads, dropout)

    def forward(
        self,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.proj_img(img_feats)
        txt = self.proj_txt(txt_feats)

        kv_pad = None if txt_mask is None else (txt_mask == 0)

        img_refined = self.img_attends_txt(img, txt, kv_padding_mask=kv_pad)
        txt_refined = self.txt_attends_img(txt, img, kv_padding_mask=None)

        fused_seq = torch.cat([img_refined, txt_refined], dim=1)

        img_pooled = img_refined.mean(dim=1)
        txt_pooled = _masked_mean(txt_refined, txt_mask)
        fused_pooled = (img_pooled + txt_pooled) * 0.5

        return fused_seq, fused_pooled


class ElementwiseFusion(nn.Module):
    """Element-wise multiplicative fusion (ablation)."""

    def __init__(self, d_img: int, d_txt: int, d_fused: int = 512) -> None:
        super().__init__()
        self.d_fused = d_fused
        self.proj_img = nn.Linear(d_img, d_fused)
        self.proj_txt = nn.Linear(d_txt, d_fused)

    def forward(
        self,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor]:
        img_pooled = self.proj_img(img_feats.mean(dim=1))
        txt_pooled = self.proj_txt(_masked_mean(txt_feats, txt_mask))
        return None, img_pooled * txt_pooled


class ConcatFusion(nn.Module):
    """Concatenation-then-project fusion (ablation)."""

    def __init__(self, d_img: int, d_txt: int, d_fused: int = 512) -> None:
        super().__init__()
        self.d_fused = d_fused
        self.proj = nn.Linear(d_img + d_txt, d_fused)

    def forward(
        self,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor]:
        img_pooled = img_feats.mean(dim=1)
        txt_pooled = _masked_mean(txt_feats, txt_mask)
        return None, self.proj(torch.cat([img_pooled, txt_pooled], dim=-1))


def build_fusion(
    fusion_type: FusionType,
    d_img: int,
    d_txt: int,
    d_fused: int = 512,
    n_heads: int = 8,
    dropout: float = 0.1,
) -> nn.Module:
    """Construct a fusion module by name."""
    if fusion_type == "co_attention":
        return CoAttentionFusion(
            d_img=d_img, d_txt=d_txt, d_fused=d_fused,
            n_heads=n_heads, dropout=dropout,
        )
    if fusion_type == "elementwise":
        return ElementwiseFusion(d_img=d_img, d_txt=d_txt, d_fused=d_fused)
    if fusion_type == "concat":
        return ConcatFusion(d_img=d_img, d_txt=d_txt, d_fused=d_fused)
    raise ValueError(
        f"unknown fusion_type {fusion_type!r}; expected one of "
        "'co_attention', 'elementwise', 'concat'"
    )
