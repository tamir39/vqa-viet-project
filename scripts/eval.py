"""Evaluate a trained config (A1 / A2 / B1 / B2) on the FoodLensVN test split.

Usage::

    uv run python scripts/eval.py --config configs/A1.yaml --checkpoint reports/A1/checkpoints/best.pt
    uv run python scripts/eval.py --config configs/B1.yaml                 # zero-shot, no checkpoint
    uv run python scripts/eval.py --config configs/B2.yaml --adapter reports/B2/adapter

Pipeline:
  1. Load config + (optional) checkpoint/adapter.
  2. Generate predictions for every test row.
  3. Apply ``normalize_answer`` to predictions and gold answers.
  4. Compute aggregate metrics (overall + by_type + by_difficulty).
  5. Write ``reports/<name>_metrics.json`` and ``reports/<name>_errors.json``.

BERTScore is on by default (xlm-roberta-base). LLM-judge is off by default —
pass ``--llm-judge`` to enable, or ``LLM_JUDGE_DISABLE=1`` to force off.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader.answer_tokenizer import AnswerTokenizer
from src.models.encoders.image_encoder import build_image_transform
from src.models.encoders.text_encoder import encode_questions, load_phobert_tokenizer
from src.utils.metrics.aggregate import aggregate_metrics
from src.utils.vn_text import normalize_answer

import scripts.train as train_mod
import scripts.infer as infer_mod


ERROR_LIMIT = 200


def _load_test_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    with Path(cfg["dataset"]["test_path"]).open("r", encoding="utf-8") as f:
        return json.load(f)


def _predict_modular(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    checkpoint: str,
    device: torch.device,
) -> list[str]:
    answer_tokenizer = AnswerTokenizer.load(cfg["dataset"]["answer_vocab"])
    model = infer_mod._build_model(cfg, answer_tokenizer).to(device).eval()

    print(f"loading checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)

    eval_tf = build_image_transform(model.image_encoder, is_training=False)
    phobert_tokenizer = load_phobert_tokenizer(cfg["model"]["text_encoder"]["model_id"])
    images_root = Path(cfg["dataset"]["images_dir"])
    max_q = cfg["loader"].get("max_question_length", 64)

    preds: list[str] = []
    for row in tqdm(rows, desc="modular infer"):
        with Image.open(images_root / row["image"]) as im:
            pixel_values = eval_tf(im.convert("RGB")).unsqueeze(0).to(device)
        tok = encode_questions(phobert_tokenizer, [row["question"]], max_length=max_q)
        with torch.no_grad():
            ids = model.generate(
                pixel_values,
                tok["input_ids"].to(device),
                tok["attention_mask"].to(device),
            )
        preds.append(answer_tokenizer.decode(ids[0].tolist()))
    return preds


def _predict_qwen(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    adapter: str | None,
) -> list[str]:
    from src.models.multimodal.qwen_vl import generate, load_qwen_vl

    md_cfg = cfg["model"]
    model, processor = load_qwen_vl(
        model_id=md_cfg.get("model_id", "Qwen/Qwen2-VL-2B-Instruct"),
        quantize_4bit=md_cfg.get("quantize_4bit", True),
    )
    if adapter:
        from peft import PeftModel
        print(f"loading LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    images_root = Path(cfg["dataset"]["images_dir"])
    preds: list[str] = []
    for row in tqdm(rows, desc=f"{md_cfg['track']} infer"):
        with Image.open(images_root / row["image"]) as im:
            preds.append(generate(model, processor, im.convert("RGB"), row["question"]))
    return preds


def _write_outputs(
    output_dir: Path,
    name: str,
    rows: list[dict[str, Any]],
    preds_norm: list[str],
    golds_norm: list[str],
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{name}_metrics.json"
    errors_path = output_dir / f"{name}_errors.json"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    errors: list[dict[str, Any]] = []
    for row, p, g in zip(rows, preds_norm, golds_norm):
        if p != g:
            errors.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "difficulty": row["difficulty"],
                    "dish": row["dish"],
                    "question": row["question"],
                    "gold": g,
                    "pred": p,
                }
            )
            if len(errors) >= ERROR_LIMIT:
                break

    with errors_path.open("w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    print(f"\nmetrics -> {metrics_path}")
    print(f"errors  -> {errors_path}  ({len(errors)} rows, capped at {ERROR_LIMIT})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="modular: path to .pt")
    parser.add_argument("--adapter", default=None, help="qwen_lora: path to LoRA adapter dir")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="where to write <name>_metrics.json + <name>_errors.json",
    )
    args = parser.parse_args()

    cfg = train_mod._load_config(Path(args.config))
    name = cfg.get("name", Path(args.config).stem)
    track = cfg["model"]["track"]
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"eval: name={name} track={track} device={device}")

    rows = _load_test_rows(cfg)
    print(f"test rows: {len(rows)}")

    t0 = time.time()
    if track == "modular":
        if not args.checkpoint:
            sys.exit("modular eval requires --checkpoint")
        preds = _predict_modular(cfg, rows, args.checkpoint, device)
    elif track == "qwen_zeroshot":
        preds = _predict_qwen(cfg, rows, adapter=None)
    elif track == "qwen_lora":
        if not args.adapter:
            sys.exit("qwen_lora eval requires --adapter")
        preds = _predict_qwen(cfg, rows, adapter=args.adapter)
    else:
        sys.exit(f"unknown model.track {track!r}")
    elapsed = time.time() - t0
    print(f"inference done in {elapsed / 60:.1f} min ({elapsed / max(1, len(rows)):.2f} s/row)")

    preds_norm = [normalize_answer(p) for p in preds]
    golds_norm = [normalize_answer(r["answer"]) for r in rows]
    types = [r["type"] for r in rows]
    diffs = [r["difficulty"] for r in rows]

    print("computing metrics ...")
    metrics = aggregate_metrics(
        preds_norm,
        golds_norm,
        types,
        diffs,
        use_bertscore=not args.no_bertscore,
        use_llm_judge=args.llm_judge,
    )
    metrics["meta"] = {
        "config": str(args.config),
        "name": name,
        "track": track,
        "checkpoint": args.checkpoint,
        "adapter": args.adapter,
        "n_test": len(rows),
        "inference_seconds": elapsed,
    }

    _write_outputs(
        Path(args.reports_dir), name, rows, preds_norm, golds_norm, metrics
    )

    o = metrics["overall"]
    print(
        f"\noverall: exact={o.get('exact', 0):.3f} soft={o.get('soft', 0):.3f} "
        f"bleu={o.get('bleu', 0):.3f} rougeL={o.get('rouge_l', 0):.3f} "
        f"meteor={o.get('meteor', 0):.3f}"
        + (f" bertF1={o['bertscore_f1']:.3f}" if "bertscore_f1" in o else "")
        + (f" judge={o['llm_judge']:.3f}" if "llm_judge" in o else "")
    )


if __name__ == "__main__":
    main()
