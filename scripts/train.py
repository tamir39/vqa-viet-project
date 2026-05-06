"""Train entry-point for FoodLensVN configs.

Usage::

    uv run python scripts/train.py --config configs/A1.yaml
    uv run python scripts/train.py --config configs/A2.yaml

Loads ``configs/base_config.yaml`` and deep-merges the per-config YAML on top,
then dispatches on ``model.track``:

* ``modular``       — A1/A2: encoders + co-attention fusion + LSTM/Transformer
                       decoder, trained via :class:`ModularTrainer`.
* ``qwen_zeroshot`` — B1: no training; placeholder until Phase 3.
* ``qwen_lora``     — B2: LoRA SFT; placeholder until Phase 3.

Outputs land under ``logging.output_dir`` (default ``reports/<config-name>/``)
with ``checkpoints/best.pt``, ``checkpoints/last.pt``, ``history.json``, and a
snapshot of the merged config as ``config.yaml``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader.answer_tokenizer import AnswerTokenizer
from src.data_loader.augment import build_image_train_transform
from src.data_loader.collate import make_collate_fn
from src.data_loader.vqa_dataset import VQADataset
from src.models.decoders.lstm_decoder import LSTMDecoder
from src.models.decoders.transformer_decoder import TransformerDecoder
from src.models.encoders.image_encoder import TimmImageEncoder, build_image_transform
from src.models.encoders.text_encoder import PhoBERTTextEncoder, load_phobert_tokenizer
from src.models.fusion.cross_attention import build_fusion
from src.models.modular_vqa import ModularVQA
from src.trainer import ModularTrainer, TrainerConfig


BASE_CONFIG = ROOT / "configs" / "base_config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _load_config(path: Path) -> dict[str, Any]:
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    with path.open("r", encoding="utf-8") as f:
        override = yaml.safe_load(f) or {}
    cfg = _deep_merge(base, override)
    cfg.setdefault("name", path.stem)
    return cfg


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_decoder(cfg_decoder: dict, vocab_size: int, d_fused: int) -> torch.nn.Module:
    dtype = cfg_decoder["type"]
    if dtype == "lstm":
        return LSTMDecoder(
            vocab_size=vocab_size,
            d_fused=d_fused,
            d_embed=cfg_decoder.get("d_embed", 256),
            d_hidden=cfg_decoder.get("d_hidden", 512),
            num_layers=cfg_decoder.get("num_layers", 1),
            dropout=cfg_decoder.get("dropout", 0.1),
        )
    if dtype == "transformer":
        return TransformerDecoder(
            vocab_size=vocab_size,
            d_fused=d_fused,
            d_model=cfg_decoder.get("d_model", 512),
            n_heads=cfg_decoder.get("n_heads", 8),
            n_layers=cfg_decoder.get("n_layers", 2),
            dim_feedforward=cfg_decoder.get("dim_feedforward", 2048),
            dropout=cfg_decoder.get("dropout", 0.1),
        )
    raise ValueError(f"unknown decoder.type {dtype!r}; expected 'lstm' or 'transformer'")


def _train_modular(cfg: dict[str, Any]) -> None:
    ds_cfg = cfg["dataset"]
    ld_cfg = cfg["loader"]
    md_cfg = cfg["model"]
    tr_cfg = cfg["training"]
    output_dir = Path(cfg["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_tokenizer = AnswerTokenizer.load(ds_cfg["answer_vocab"])
    print(f"answer vocab size: {answer_tokenizer.vocab_size}")

    print("building encoders ...")
    image_encoder = TimmImageEncoder(
        model_id=md_cfg["image_encoder"]["model_id"],
        pretrained=md_cfg["image_encoder"].get("pretrained", True),
        freeze=md_cfg["image_encoder"].get("freeze", True),
        unfreeze_last_n=md_cfg["image_encoder"].get("unfreeze_last_n", 0),
    )
    text_encoder = PhoBERTTextEncoder(
        model_id=md_cfg["text_encoder"]["model_id"],
        freeze=md_cfg["text_encoder"].get("freeze", True),
        unfreeze_last_n=md_cfg["text_encoder"].get("unfreeze_last_n", 0),
    )
    phobert_tokenizer = load_phobert_tokenizer(md_cfg["text_encoder"]["model_id"])

    fusion = build_fusion(
        fusion_type=md_cfg["fusion"]["type"],
        d_img=image_encoder.feature_dim,
        d_txt=text_encoder.hidden_size,
        d_fused=md_cfg["fusion"]["d_fused"],
        n_heads=md_cfg["fusion"].get("n_heads", 8),
        dropout=md_cfg["fusion"].get("dropout", 0.1),
    )
    decoder = _build_decoder(
        md_cfg["decoder"],
        vocab_size=answer_tokenizer.vocab_size,
        d_fused=md_cfg["fusion"]["d_fused"],
    )
    model = ModularVQA(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        fusion=fusion,
        decoder=decoder,
        decoder_input=md_cfg["decoder_input"],
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"model: {n_train / 1e6:.2f}M trainable / {n_total / 1e6:.2f}M total")

    eval_tf = build_image_transform(image_encoder, is_training=False)
    train_tf = (
        build_image_train_transform(eval_tf)
        if ld_cfg.get("augment_train", True)
        else eval_tf
    )

    print("loading splits ...")
    train_ds = VQADataset(
        rows_path=ds_cfg["train_path"],
        images_root=ds_cfg["images_dir"],
        image_transform=train_tf,
        answer_tokenizer=answer_tokenizer,
    )
    val_ds = VQADataset(
        rows_path=ds_cfg["val_path"],
        images_root=ds_cfg["images_dir"],
        image_transform=eval_tf,
        answer_tokenizer=answer_tokenizer,
    )
    print(f"  train={len(train_ds)} val={len(val_ds)}")

    collate = make_collate_fn(
        phobert_tokenizer,
        max_question_length=ld_cfg.get("max_question_length", 64),
    )
    pin = ld_cfg.get("pin_memory", True) and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=ld_cfg["batch_size"],
        shuffle=True,
        num_workers=ld_cfg.get("num_workers", 2),
        pin_memory=pin,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ld_cfg["batch_size"],
        shuffle=False,
        num_workers=ld_cfg.get("num_workers", 2),
        pin_memory=pin,
        collate_fn=collate,
    )

    trainer_cfg = TrainerConfig(
        output_dir=str(output_dir),
        epochs=tr_cfg.get("epochs", 10),
        lr=tr_cfg.get("lr", 5.0e-5),
        weight_decay=tr_cfg.get("weight_decay", 0.01),
        warmup_ratio=tr_cfg.get("warmup_ratio", 0.05),
        grad_clip=tr_cfg.get("grad_clip", 1.0),
        amp=tr_cfg.get("amp", True),
        log_every=tr_cfg.get("log_every", 50),
    )
    trainer = ModularTrainer(model, train_loader, val_loader, trainer_cfg)

    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    result = trainer.fit()
    print(f"\ndone. best_val_loss={result['best_val_loss']:.4f}")
    print(f"checkpoints: {output_dir / 'checkpoints'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--config", required=True, help="path to per-config YAML (e.g., configs/A1.yaml)"
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    _seed_all(seed)
    print(f"config: {cfg_path}  name={cfg.get('name')}  seed={seed}")

    track = cfg["model"]["track"]
    if track == "modular":
        _train_modular(cfg)
    elif track in ("qwen_zeroshot", "qwen_lora"):
        sys.exit(
            f"track={track!r} dispatch is Phase-3 work (see TASKS.md §3.2/§3.3); "
            "scripts/train.py only handles 'modular' so far."
        )
    else:
        sys.exit(f"unknown model.track {track!r}")


if __name__ == "__main__":
    main()
