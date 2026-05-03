"""Image encoder for the modular VQA pipeline (A1/A2).

Wraps a ``timm`` backbone and exposes its spatial features as a uniform
``(B, P, D)`` patch sequence that the co-attention fusion module can attend
over. Two backbones are supported out of the box:

* ``resnet50`` (default) — CNN; ``forward_features`` returns ``(B, C, H, W)``
  which we flatten to ``(B, H*W, C)``. For 224×224 input this is ``(B, 49, 2048)``.
* ``vit_small_patch16_224`` (ablation) — ViT; ``forward_features`` returns
  ``(B, 1+N, D)`` with the CLS token at position 0. We strip CLS and return
  ``(B, N, D)`` so the fusion module sees a homogeneous patch sequence
  regardless of backbone family.

The wrapper also exposes the timm-resolved image transform (mean/std/resize)
so the dataset layer can preprocess identically to what the backbone was
pretrained on.

Set ``KAGGLE_NO_INTERNET=1`` to disable timm's HF Hub fallback. timm will then
load weights only from its local cache; pre-stage models on Kaggle.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn


DEFAULT_MODEL_ID: str = "resnet50"


def _local_files_only() -> bool:
    return os.environ.get("KAGGLE_NO_INTERNET") == "1"


class TimmImageEncoder(nn.Module):
    """timm backbone returning patch features ``(B, P, D)``.

    Args:
        model_id: timm model name. Defaults to ``resnet50``.
            Recommended ablation: ``vit_small_patch16_224``.
        pretrained: Load ImageNet pretrained weights. Set to ``False`` only
            for unit tests.
        freeze: Freeze all parameters when ``True`` (default).
        unfreeze_last_n: Number of trailing blocks (CNN ``layer4`` sub-blocks
            or ViT transformer blocks) to unfreeze when ``freeze=True``.
            Ignored when ``freeze=False``.

    Attributes:
        feature_dim: Channel/embedding dimension ``D`` of the output patches.
        is_vit: Whether the backbone is a ViT-family model (CLS-token aware).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        pretrained: bool = True,
        freeze: bool = True,
        unfreeze_last_n: int = 0,
    ) -> None:
        super().__init__()
        import timm

        self.model_id = model_id

        if _local_files_only() and pretrained:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self.backbone = timm.create_model(
            model_id,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        self.is_vit = "vit" in model_id.lower() or "deit" in model_id.lower()
        self.feature_dim: int = int(self.backbone.num_features)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_last_n > 0:
                self._unfreeze_last_n_blocks(unfreeze_last_n)

    def _unfreeze_last_n_blocks(self, n: int) -> None:
        if self.is_vit and hasattr(self.backbone, "blocks"):
            blocks = self.backbone.blocks
        elif hasattr(self.backbone, "layer4"):
            blocks = self.backbone.layer4
        else:
            return
        n = min(n, len(blocks))
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return patch features ``(B, P, D)``.

        ViT outputs have the CLS token at position 0 stripped before return,
        so the caller always sees a pure patch sequence.
        """
        feats = self.backbone.forward_features(pixel_values)

        if feats.dim() == 4:
            B, C, H, W = feats.shape
            return feats.flatten(2).transpose(1, 2).contiguous()

        if feats.dim() == 3:
            return feats[:, 1:, :].contiguous() if self.is_vit else feats

        raise RuntimeError(
            f"unexpected forward_features rank {feats.dim()} for {self.model_id!r}"
        )


def build_image_transform(
    encoder: TimmImageEncoder,
    is_training: bool = False,
) -> Any:
    """Return the timm-resolved torchvision transform for the wrapped backbone."""
    import timm

    data_config = timm.data.resolve_model_data_config(encoder.backbone)
    return timm.data.create_transform(**data_config, is_training=is_training)
