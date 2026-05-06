"""Single-example inference for a trained modular config.

Usage::

    uv run python scripts/infer.py \
        --config configs/A1.yaml \
        --checkpoint reports/A1/checkpoints/best.pt \
        --image data/foodlensvn/images/squared/test/pho_001.jpg \
        --question "Đây là món gì?"

Mirrors the model-build path in :mod:`scripts.train` (so checkpoint shapes
match), loads the saved ``state_dict``, runs the encoder→fusion→decoder pass,
and prints the decoded answer.

Phase-3 tracks (``qwen_zeroshot``, ``qwen_lora``) are not handled here yet.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader.answer_tokenizer import AnswerTokenizer
from src.models.encoders.image_encoder import TimmImageEncoder, build_image_transform
from src.models.encoders.text_encoder import (
    PhoBERTTextEncoder,
    encode_questions,
    load_phobert_tokenizer,
)
from src.models.fusion.cross_attention import build_fusion
from src.models.modular_vqa import ModularVQA

# Reuse the train-script helpers so the build path stays in one place.
import scripts.train as train_mod


def _build_model(cfg: dict[str, Any], answer_tokenizer: AnswerTokenizer) -> ModularVQA:
    md = cfg["model"]
    image_encoder = TimmImageEncoder(
        model_id=md["image_encoder"]["model_id"],
        pretrained=md["image_encoder"].get("pretrained", True),
        freeze=True,
    )
    text_encoder = PhoBERTTextEncoder(
        model_id=md["text_encoder"]["model_id"], freeze=True
    )
    fusion = build_fusion(
        fusion_type=md["fusion"]["type"],
        d_img=image_encoder.feature_dim,
        d_txt=text_encoder.hidden_size,
        d_fused=md["fusion"]["d_fused"],
        n_heads=md["fusion"].get("n_heads", 8),
        dropout=md["fusion"].get("dropout", 0.1),
    )
    decoder = train_mod._build_decoder(
        md["decoder"],
        vocab_size=answer_tokenizer.vocab_size,
        d_fused=md["fusion"]["d_fused"],
    )
    return ModularVQA(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        fusion=fusion,
        decoder=decoder,
        decoder_input=md["decoder_input"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--device", default=None, help="cuda|cpu (default: cuda if available)"
    )
    args = parser.parse_args()

    cfg = train_mod._load_config(Path(args.config))
    track = cfg["model"]["track"]
    if track != "modular":
        sys.exit(
            f"track={track!r} inference is not implemented yet; this script handles 'modular' only."
        )

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    answer_tokenizer = AnswerTokenizer.load(cfg["dataset"]["answer_vocab"])
    model = _build_model(cfg, answer_tokenizer).to(device).eval()

    print(f"loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)

    image_encoder = model.image_encoder  # type: ignore[assignment]
    eval_tf = build_image_transform(image_encoder, is_training=False)
    with Image.open(args.image) as im:
        pixel_values = eval_tf(im.convert("RGB")).unsqueeze(0).to(device)

    phobert_tokenizer = load_phobert_tokenizer(cfg["model"]["text_encoder"]["model_id"])
    tok = encode_questions(
        phobert_tokenizer,
        [args.question],
        max_length=cfg["loader"].get("max_question_length", 64),
    )
    input_ids = tok["input_ids"].to(device)
    attn_mask = tok["attention_mask"].to(device)

    with torch.no_grad():
        ids = model.generate(pixel_values, input_ids, attn_mask)

    answer = answer_tokenizer.decode(ids[0].tolist())
    print(f"\nQ: {args.question}")
    print(f"A: {answer}")


if __name__ == "__main__":
    main()
