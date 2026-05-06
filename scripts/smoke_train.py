"""End-to-end pre-flight check for A1 + A2 (CPU, 4 rows, 1 step).

Doesn't write anything to disk; just verifies that the full graph
``encoder -> fusion -> decoder -> CE loss -> backward -> step`` works for
both configs without shape mismatches. Run before committing to a Kaggle GPU
hour.

Usage::

    uv run python scripts/smoke_train.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train as train_mod
from src.data_loader.answer_tokenizer import AnswerTokenizer
from src.data_loader.collate import make_collate_fn
from src.data_loader.vqa_dataset import VQADataset
from src.models.encoders.image_encoder import TimmImageEncoder, build_image_transform
from src.models.encoders.text_encoder import PhoBERTTextEncoder, load_phobert_tokenizer
from src.models.fusion.cross_attention import build_fusion
from src.models.modular_vqa import ModularVQA


def smoke_one(config_path: str) -> None:
    cfg = train_mod._load_config(Path(config_path))
    print(f"\n=== {cfg.get('name', config_path)} ===  decoder_input={cfg['model']['decoder_input']}")

    answer_tok = AnswerTokenizer.load(cfg["dataset"]["answer_vocab"])
    md = cfg["model"]
    image_enc = TimmImageEncoder(
        model_id=md["image_encoder"]["model_id"], pretrained=False, freeze=False
    )
    text_enc = PhoBERTTextEncoder(model_id=md["text_encoder"]["model_id"], freeze=False)
    phobert_tok = load_phobert_tokenizer(md["text_encoder"]["model_id"])

    fusion = build_fusion(
        fusion_type=md["fusion"]["type"],
        d_img=image_enc.feature_dim,
        d_txt=text_enc.hidden_size,
        d_fused=md["fusion"]["d_fused"],
        n_heads=md["fusion"].get("n_heads", 8),
        dropout=md["fusion"].get("dropout", 0.1),
    )
    decoder = train_mod._build_decoder(
        md["decoder"], vocab_size=answer_tok.vocab_size, d_fused=md["fusion"]["d_fused"]
    )
    model = ModularVQA(
        image_encoder=image_enc,
        text_encoder=text_enc,
        fusion=fusion,
        decoder=decoder,
        decoder_input=md["decoder_input"],
    ).train()

    eval_tf = build_image_transform(image_enc, is_training=False)
    full_ds = VQADataset(
        rows_path=cfg["dataset"]["train_path"],
        images_root=cfg["dataset"]["images_dir"],
        image_transform=eval_tf,
        answer_tokenizer=answer_tok,
    )
    sub = Subset(full_ds, [0, 1, 2, 3])
    loader = DataLoader(
        sub, batch_size=4, shuffle=False,
        collate_fn=make_collate_fn(phobert_tok, max_question_length=32),
    )
    batch = next(iter(loader))
    print(
        f"batch shapes: pixel={tuple(batch['pixel_values'].shape)} "
        f"q_ids={tuple(batch['question_input_ids'].shape)} "
        f"answer_ids={tuple(batch['answer_ids'].shape)}"
    )

    target = batch["answer_ids"]
    decoder_in = target[:, :-1].contiguous()
    labels = target[:, 1:].contiguous()

    logits = model(
        pixel_values=batch["pixel_values"],
        question_input_ids=batch["question_input_ids"],
        question_attention_mask=batch["question_attention_mask"],
        target_ids=decoder_in,
    )
    print(f"logits: {tuple(logits.shape)}  expected (B=4, T-1={target.size(1) - 1}, V={answer_tok.vocab_size})")
    assert logits.shape == (4, target.size(1) - 1, answer_tok.vocab_size)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=AnswerTokenizer.PAD_ID)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    print(f"loss: {loss.item():.4f}  finite={torch.isfinite(loss).item()}")
    assert torch.isfinite(loss)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("backward + step ok")

    model.eval()
    with torch.no_grad():
        ids = model.generate(
            batch["pixel_values"],
            batch["question_input_ids"],
            batch["question_attention_mask"],
        )
    print(f"generate: {tuple(ids.shape)}  decoded[0]={answer_tok.decode(ids[0].tolist())!r}")


def main() -> None:
    smoke_one("configs/A1.yaml")
    smoke_one("configs/A2.yaml")
    print("\nall pre-flight checks passed.")


if __name__ == "__main__":
    main()
