"""Smoke test for the modular VQA encoders (TASKS §2.1).

Loads :class:`PhoBERTTextEncoder` and :class:`TimmImageEncoder` on CPU with
random tensor inputs and asserts their output shapes match the contract the
fusion module expects.

PhoBERT weights (~500 MB) and the timm backbone (~100 MB for ``resnet50``) are
downloaded on first run via the HuggingFace / timm caches. Set
``KAGGLE_NO_INTERNET=1`` to force local-cache-only loads.

Usage::

    uv run python scripts/smoke_encoders.py
    uv run python scripts/smoke_encoders.py --image-model vit_small_patch16_224
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.encoders.image_encoder import TimmImageEncoder, build_image_transform
from src.models.encoders.text_encoder import (
    PhoBERTTextEncoder,
    encode_questions,
    load_phobert_tokenizer,
)


SAMPLE_QUESTIONS = [
    "Đây là món gì?",
    "Món này có bao nhiêu nguyên liệu chính?",
]


def smoke_text_encoder() -> tuple[int, int, int]:
    print("[text] loading PhoBERTTextEncoder ...")
    enc = PhoBERTTextEncoder(freeze=True).eval()
    tokenizer = load_phobert_tokenizer()
    batch = encode_questions(tokenizer, SAMPLE_QUESTIONS, max_length=32)

    with torch.no_grad():
        out = enc(batch["input_ids"], batch["attention_mask"])

    B, T, D = out.shape
    assert B == len(SAMPLE_QUESTIONS), f"batch mismatch: {B} vs {len(SAMPLE_QUESTIONS)}"
    assert D == enc.hidden_size, f"hidden dim mismatch: {D} vs {enc.hidden_size}"
    assert out.dtype == torch.float32
    print(f"[text] OK  shape=(B={B}, T={T}, D={D})  hidden_size={enc.hidden_size}")
    return B, T, D


def smoke_image_encoder(model_id: str) -> tuple[int, int, int]:
    print(f"[image] loading TimmImageEncoder({model_id!r}) ...")
    enc = TimmImageEncoder(model_id=model_id, pretrained=True, freeze=True).eval()
    transform = build_image_transform(enc, is_training=False)

    H, W = 224, 224
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            if hasattr(t, "size"):
                size = t.size
                if isinstance(size, (tuple, list)) and len(size) == 2:
                    H, W = int(size[0]), int(size[1])
                elif isinstance(size, int):
                    H = W = int(size)
                break

    B = 2
    pixel_values = torch.randn(B, 3, H, W)
    with torch.no_grad():
        out = enc(pixel_values)

    Bo, P, D = out.shape
    assert Bo == B, f"batch mismatch: {Bo} vs {B}"
    assert P > 0, f"empty patch dim: {P}"
    assert D == enc.feature_dim, f"feature dim mismatch: {D} vs {enc.feature_dim}"
    print(f"[image] OK shape=(B={Bo}, P={P}, D={D})  feature_dim={enc.feature_dim}  is_vit={enc.is_vit}")
    return Bo, P, D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--image-model",
        default="resnet50",
        help="timm model id (default: resnet50; ablation: vit_small_patch16_224)",
    )
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    args = parser.parse_args()

    if not args.skip_text:
        smoke_text_encoder()
    if not args.skip_image:
        smoke_image_encoder(args.image_model)
    print("\nall encoder smoke checks passed.")


if __name__ == "__main__":
    main()
