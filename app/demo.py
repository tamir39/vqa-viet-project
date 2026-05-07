"""Gradio demo: A1 / A2 / B1 / B2 side-by-side VQA on a single image + question.

Each track is lazy-loaded on first request and cached, so app startup is fast
even when the underlying weights are large. Tracks whose checkpoint/adapter is
missing on disk are skipped (the column is shown as ``[skipped]`` instead).

Designed to run inside a Kaggle notebook with internet on. Launch with::

    uv run python app/demo.py --share

Other flags::

    --a1-checkpoint reports/A1/checkpoints/best.pt
    --a2-checkpoint reports/A2/checkpoints/best.pt
    --b2-adapter    reports/B2/adapter
    --device        cuda|cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_modular_predictor(
    config_path: Path,
    checkpoint: Path,
    device_str: str,
) -> Callable[[Any, str], str]:
    import torch
    from PIL import Image

    from src.data_loader.answer_tokenizer import AnswerTokenizer
    from src.models.encoders.image_encoder import build_image_transform
    from src.models.encoders.text_encoder import (
        encode_questions,
        load_phobert_tokenizer,
    )

    import scripts.train as train_mod
    import scripts.infer as infer_mod

    cfg = train_mod._load_config(config_path)
    device = torch.device(device_str)
    answer_tokenizer = AnswerTokenizer.load(cfg["dataset"]["answer_vocab"])
    model = infer_mod._build_model(cfg, answer_tokenizer).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)

    eval_tf = build_image_transform(model.image_encoder, is_training=False)
    phobert_tokenizer = load_phobert_tokenizer(cfg["model"]["text_encoder"]["model_id"])
    max_q = cfg["loader"].get("max_question_length", 64)

    def predict(image: Image.Image, question: str) -> str:
        pixel_values = eval_tf(image.convert("RGB")).unsqueeze(0).to(device)
        tok = encode_questions(phobert_tokenizer, [question], max_length=max_q)
        with torch.no_grad():
            ids = model.generate(
                pixel_values,
                tok["input_ids"].to(device),
                tok["attention_mask"].to(device),
            )
        return answer_tokenizer.decode(ids[0].tolist())

    return predict


class _QwenRuntime:
    """Shared Qwen2-VL base for B1 and B2.

    B1 and B2 use the same NF4 base model — loading two copies blows VRAM on
    a T4. This holds one base, optionally wraps it with a LoRA adapter, and
    exposes B1 (adapter disabled) and B2 (adapter enabled) as predictors.
    """

    def __init__(self, b1_cfg: Path, b2_adapter: Path | None):
        from src.models.multimodal.qwen_vl import load_qwen_vl

        import scripts.train as train_mod

        cfg = train_mod._load_config(b1_cfg)
        md_cfg = cfg["model"]
        base, processor = load_qwen_vl(
            model_id=md_cfg.get("model_id", "Qwen/Qwen2-VL-2B-Instruct"),
            quantize_4bit=md_cfg.get("quantize_4bit", True),
        )
        self.processor = processor
        self.has_adapter = b2_adapter is not None and b2_adapter.exists()

        if self.has_adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, str(b2_adapter))
        else:
            self.model = base
        self.model.eval()

    def predict_b1(self, image: Any, question: str) -> str:
        from src.models.multimodal.qwen_vl import generate
        if self.has_adapter:
            with self.model.disable_adapter():
                return generate(self.model, self.processor, image.convert("RGB"), question)
        return generate(self.model, self.processor, image.convert("RGB"), question)

    def predict_b2(self, image: Any, question: str) -> str:
        from src.models.multimodal.qwen_vl import generate
        return generate(self.model, self.processor, image.convert("RGB"), question)


class _LazyTrack:
    """One column in the demo: holds builder + cached predictor."""

    def __init__(self, name: str, builder: Callable[[], Callable[[Any, str], str]] | None):
        self.name = name
        self._builder = builder
        self._predictor: Callable[[Any, str], str] | None = None

    @property
    def available(self) -> bool:
        return self._builder is not None

    def predict(self, image: Any, question: str) -> tuple[str, float]:
        if not self.available:
            return "[skipped]", 0.0
        if self._predictor is None:
            self._predictor = self._builder()  # type: ignore[misc]
        t0 = time.time()
        ans = self._predictor(image, question)
        return ans, time.time() - t0


def _build_tracks(args: argparse.Namespace, device_str: str) -> list[_LazyTrack]:
    cfg_dir = ROOT / "configs"

    def modular_builder(cfg_name: str, ckpt: Path) -> Callable[[], Callable[[Any, str], str]]:
        return lambda: _build_modular_predictor(cfg_dir / cfg_name, ckpt, device_str)

    # Shared Qwen runtime: lazy-loaded once on first B1 or B2 call.
    b2_adapter = Path(args.b2_adapter)
    qwen_holder: dict[str, _QwenRuntime] = {}

    def get_qwen() -> _QwenRuntime:
        if "rt" not in qwen_holder:
            qwen_holder["rt"] = _QwenRuntime(
                cfg_dir / "B1.yaml",
                b2_adapter if b2_adapter.exists() else None,
            )
        return qwen_holder["rt"]

    def b1_builder() -> Callable[[Any, str], str]:
        rt = get_qwen()
        return rt.predict_b1

    def b2_builder() -> Callable[[Any, str], str]:
        rt = get_qwen()
        return rt.predict_b2

    tracks: list[_LazyTrack] = []

    a1_ckpt = Path(args.a1_checkpoint)
    tracks.append(
        _LazyTrack("A1", modular_builder("A1.yaml", a1_ckpt) if a1_ckpt.exists() else None)
    )
    a2_ckpt = Path(args.a2_checkpoint)
    tracks.append(
        _LazyTrack("A2", modular_builder("A2.yaml", a2_ckpt) if a2_ckpt.exists() else None)
    )
    tracks.append(_LazyTrack("B1", b1_builder))
    tracks.append(
        _LazyTrack("B2", b2_builder if b2_adapter.exists() else None)
    )
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--a1-checkpoint", default="reports/A1/checkpoints/best.pt")
    parser.add_argument("--a2-checkpoint", default="reports/A2/checkpoints/best.pt")
    parser.add_argument("--b2-adapter", default="reports/B2/adapter")
    parser.add_argument("--device", default=None)
    parser.add_argument("--share", action="store_true", help="enable gradio share link (Kaggle)")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="warm-load every track before launching gradio so first-request latency is gone",
    )
    args = parser.parse_args()

    import torch
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"demo: device={device_str}", flush=True)

    tracks = _build_tracks(args, device_str)
    available = [t.name for t in tracks if t.available]
    skipped = [t.name for t in tracks if not t.available]
    print(f"available tracks: {available}", flush=True)
    if skipped:
        print(f"skipped (missing artifacts): {skipped}", flush=True)

    if args.eager:
        # Trigger lazy load of every available track. We have to call the
        # underlying builder rather than predict() because predict() needs an
        # image; building the predictor is what loads the weights. After this
        # loop, the first browser request hits cached models.
        for t in tracks:
            if not t.available:
                continue
            print(f"[eager] loading {t.name} ...", flush=True)
            t._predictor = t._builder()  # type: ignore[misc]
        print("[eager] all tracks warm; gradio is about to launch.", flush=True)

    import gradio as gr
    import pandas as pd

    def run(image: Any, question: str) -> pd.DataFrame:
        if image is None or not (question or "").strip():
            return pd.DataFrame(columns=["config", "answer", "seconds"])
        rows = []
        for t in tracks:
            ans, secs = t.predict(image, question)
            rows.append({"config": t.name, "answer": ans, "seconds": round(secs, 3)})
        return pd.DataFrame(rows, columns=["config", "answer", "seconds"])

    with gr.Blocks(title="FoodLensVN — VQA Demo") as demo:
        gr.Markdown(
            "## FoodLensVN — A1 / A2 / B1 / B2 side-by-side\n"
            "Upload an image of a Vietnamese dish and ask a question in Vietnamese. "
            "Each model is lazy-loaded on first use, so the first call to a column is slow."
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Image")
                question_in = gr.Textbox(
                    label="Câu hỏi (tiếng Việt)",
                    placeholder="Đây là món gì?",
                )
                submit = gr.Button("Predict", variant="primary")
            with gr.Column():
                results = gr.Dataframe(
                    headers=["config", "answer", "seconds"],
                    label="Predictions",
                    wrap=True,
                )
        submit.click(run, inputs=[image_in, question_in], outputs=[results])
        question_in.submit(run, inputs=[image_in, question_in], outputs=[results])

    demo.queue().launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
