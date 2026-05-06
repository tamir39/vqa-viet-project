"""LoRA SFT trainer for Qwen2-VL on FoodLensVN (B2 track).

NF4-quantized base + LoRA adapters on attention projections + bf16 compute.
Manual loop to avoid pulling TRL: peft alone is sufficient, and the
multimodal collation needs custom logic (image branch in ``processor``).

Saves the LoRA adapter to ``<output_dir>/adapter/`` and per-epoch metrics to
``<output_dir>/history.json``. Mirrors the artifact layout of
:mod:`src.trainer.modular_trainer` so downstream tooling (eval, demo) can
treat A/B runs uniformly.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.models.multimodal.qwen_vl import SYSTEM, build_messages, load_qwen_vl


@dataclass
class PeftTrainerConfig:
    output_dir: str
    epochs: int = 3
    lr: float = 2.0e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    batch_size: int = 2
    grad_accum_steps: int = 8
    grad_clip: float = 0.3
    gradient_checkpointing: bool = True
    amp: bool = True
    log_every: int = 25
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class _QwenVQADataset(Dataset):
    """Returns (PIL image, question, answer) — collation lives in the trainer."""

    def __init__(self, rows_path: str | Path, images_root: str | Path) -> None:
        with Path(rows_path).open("r", encoding="utf-8") as f:
            self.rows = json.load(f)
        self.images_root = Path(images_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        img = Image.open(self.images_root / r["image"]).convert("RGB")
        return {"image": img, "question": r["question"], "answer": r["answer"]}


class _QwenVQACollator:
    """Builds input_ids/pixel_values + supervised labels (mask the prompt with -100).

    For each row we tokenize the full chat (system + user + assistant=answer)
    twice: once *without* the answer to get the prompt length, once *with* the
    answer to get the supervised target. Labels for prompt positions are -100
    so cross-entropy only flows on the answer tokens.
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        full_texts: list[str] = []
        prompt_lens: list[int] = []
        images: list[Any] = []

        for row in batch:
            messages = build_messages(row["image"], row["question"])
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = prompt_text + row["answer"] + self.processor.tokenizer.eos_token

            prompt_ids = self.processor.tokenizer(prompt_text, add_special_tokens=False)[
                "input_ids"
            ]
            prompt_lens.append(len(prompt_ids))
            full_texts.append(full_text)
            images.append(row["image"])

        enc = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[input_ids == pad_id] = -100
        for i, p in enumerate(prompt_lens):
            labels[i, :p] = -100

        enc["labels"] = labels
        enc["attention_mask"] = attention_mask
        return enc


def _warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class PeftTrainer:
    """LoRA SFT loop for Qwen2-VL.

    Args:
        train_rows: path to train.json (processed).
        val_rows: path to val.json (processed).
        images_root: root for relative image paths in the JSON.
        cfg: :class:`PeftTrainerConfig`.
        model_id: HF model id; default ``Qwen/Qwen2-VL-2B-Instruct``.
        quantize_4bit: NF4 base. Disable only on CPU dry-runs.
    """

    def __init__(
        self,
        train_rows: str,
        val_rows: str,
        images_root: str,
        cfg: PeftTrainerConfig,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        quantize_4bit: bool = True,
    ) -> None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"loading {model_id} (4bit={quantize_4bit}) ...")
        model, processor = load_qwen_vl(model_id=model_id, quantize_4bit=quantize_4bit)
        if quantize_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.gradient_checkpointing
            )
        elif cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            target_modules=cfg.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        self.model = model
        self.processor = processor

        collator = _QwenVQACollator(processor)
        train_ds = _QwenVQADataset(train_rows, images_root)
        val_ds = _QwenVQADataset(val_rows, images_root)
        print(f"  train={len(train_ds)} val={len(val_ds)}")

        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,                    # processor is not always pickle-safe
            collate_fn=collator,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        device = self.model.device
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in self.val_loader:
            batch = self._move(batch)
            out = self.model(**batch)
            total_loss += float(out.loss)
            n += 1
        self.model.train()
        return total_loss / max(1, n)

    def fit(self) -> dict[str, Any]:
        cfg = self.cfg
        steps_per_epoch = math.ceil(len(self.train_loader) / cfg.grad_accum_steps)
        total_steps = steps_per_epoch * cfg.epochs
        warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(
            trainable, lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        history: list[dict[str, Any]] = []
        best_val = float("inf")
        global_step = 0
        t0 = time.time()
        self.model.train()

        for epoch in range(1, cfg.epochs + 1):
            running = 0.0
            ep_steps = 0
            optim.zero_grad(set_to_none=True)

            for i, batch in enumerate(self.train_loader):
                batch = self._move(batch)
                out = self.model(**batch)
                loss = out.loss / cfg.grad_accum_steps
                loss.backward()
                running += float(out.loss)

                if (i + 1) % cfg.grad_accum_steps == 0 or (i + 1) == len(self.train_loader):
                    if cfg.grad_clip:
                        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                    lr_scale = _warmup_cosine(global_step, total_steps, warmup_steps)
                    for g in optim.param_groups:
                        g["lr"] = cfg.lr * lr_scale
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1
                    ep_steps += 1

                    if global_step % cfg.log_every == 0:
                        print(
                            f"  epoch {epoch} step {global_step}/{total_steps} "
                            f"loss={running / max(1, i + 1):.4f} lr={cfg.lr * lr_scale:.2e}"
                        )

            train_loss = running / max(1, len(self.train_loader))
            val_loss = self._evaluate()
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )

            if val_loss < best_val:
                best_val = val_loss
                self._save_adapter("adapter")
                print(f"  new best val_loss={best_val:.4f}; adapter saved")

        elapsed = time.time() - t0
        result = {
            "best_val_loss": best_val,
            "epochs": cfg.epochs,
            "total_seconds": elapsed,
            "history": history,
        }
        with (self.output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\ntraining done in {elapsed / 60:.1f} min; best val_loss={best_val:.4f}")
        return result

    def _save_adapter(self, name: str) -> None:
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.processor.save_pretrained(str(path))
