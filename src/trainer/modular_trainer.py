"""Training loop for the modular VQA pipeline (A1/A2).

Trains a :class:`~src.models.modular_vqa.ModularVQA` over the FoodLensVN
processed splits with:

* AMP (bf16 if supported, else fp16) via ``torch.amp``.
* AdamW + warmup → cosine LR schedule (linear warmup, cosine decay to 0).
* Gradient clipping (default ``max_norm=1.0``).
* Teacher-forced cross-entropy with PAD ignored
  (decoder input: ``target_ids[:, :-1]``, labels: ``target_ids[:, 1:]``).
* Best-by-val-loss checkpoint saved to
  ``<output_dir>/checkpoints/best.pt`` plus ``last.pt`` every epoch.

The trainer is decoder-agnostic — A1 (LSTM, ``decoder_input='pooled'``) and A2
(Transformer, ``decoder_input='seq'``) share this loop unchanged.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data_loader.answer_tokenizer import AnswerTokenizer


@dataclass
class TrainerConfig:
    output_dir: str
    epochs: int = 10
    lr: float = 5.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    log_every: int = 50
    save_last: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def _pick_amp_dtype() -> torch.dtype | None:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return None


def _warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return float(step) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class ModularTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: TrainerConfig,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.output_dir = Path(self.cfg.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.amp_dtype = _pick_amp_dtype() if self.cfg.amp else None
        self.use_scaler = self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("model has no trainable parameters")
        self.optimizer = AdamW(
            trainable, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        self.total_steps = max(1, self.cfg.epochs * len(self.train_loader))
        self.warmup_steps = max(1, int(self.cfg.warmup_ratio * self.total_steps))

        self.criterion = nn.CrossEntropyLoss(ignore_index=AnswerTokenizer.PAD_ID)
        self.history: list[dict[str, Any]] = []
        self.best_val_loss: float = float("inf")
        self.global_step: int = 0

    def _set_lr(self, step: int) -> float:
        scale = _warmup_cosine(step, self.total_steps, self.warmup_steps)
        lr = self.cfg.lr * scale
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _step_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        target = batch["answer_ids"]
        decoder_in = target[:, :-1].contiguous()
        labels = target[:, 1:].contiguous()
        logits = self.model(
            pixel_values=batch["pixel_values"],
            question_input_ids=batch["question_input_ids"],
            question_attention_mask=batch["question_attention_mask"],
            target_ids=decoder_in,
        )
        # logits: (B, T-1, V)
        return self.criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for it, raw in enumerate(self.train_loader):
            batch = self._move(raw)
            lr = self._set_lr(self.global_step)

            self.optimizer.zero_grad(set_to_none=True)
            if self.amp_dtype is not None:
                with torch.amp.autocast(self.device.type, dtype=self.amp_dtype):
                    loss = self._step_loss(batch)
            else:
                loss = self._step_loss(batch)

            if self.use_scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.grad_clip,
                )
                self.optimizer.step()

            running += loss.item() * batch["pixel_values"].size(0)
            n += batch["pixel_values"].size(0)
            self.global_step += 1

            if (it + 1) % self.cfg.log_every == 0:
                avg = running / max(1, n)
                print(
                    f"  epoch {epoch} step {it + 1}/{len(self.train_loader)} "
                    f"loss={avg:.4f} lr={lr:.2e}"
                )

        return running / max(1, n)

    @torch.no_grad()
    def _validate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        running, n = 0.0, 0
        for raw in self.val_loader:
            batch = self._move(raw)
            if self.amp_dtype is not None:
                with torch.amp.autocast(self.device.type, dtype=self.amp_dtype):
                    loss = self._step_loss(batch)
            else:
                loss = self._step_loss(batch)
            running += loss.item() * batch["pixel_values"].size(0)
            n += batch["pixel_values"].size(0)
        return running / max(1, n)

    def _save_checkpoint(self, name: str, epoch: int, val_loss: float) -> Path:
        path = self.ckpt_dir / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
                "val_loss": val_loss,
                "config": asdict(self.cfg),
            },
            path,
        )
        return path

    def fit(self) -> dict[str, Any]:
        print(
            f"trainer: device={self.device} amp_dtype={self.amp_dtype} "
            f"steps={self.total_steps} warmup={self.warmup_steps}"
        )
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate()
            dt = time.time() - t0

            entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time_s": dt,
            }
            self.history.append(entry)
            print(
                f"epoch {epoch}/{self.cfg.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"time={dt:.1f}s"
            )

            if self.cfg.save_last:
                self._save_checkpoint("last.pt", epoch, val_loss)
            if math.isfinite(val_loss) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self._save_checkpoint("best.pt", epoch, val_loss)
                print(f"  ↳ new best (val_loss={val_loss:.4f}) saved to {best_path}")

        with (self.output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {"best_val_loss": self.best_val_loss, "history": self.history}
