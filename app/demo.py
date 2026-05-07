"""Gradio demo: A1 / A2 / B1 / B2 side-by-side VQA on Vietnamese dishes.

Layout: image + question on the left, four model cards on the right (2x2).
Each card shows the architecture and the model's prediction + latency.

Memory-conscious loading:
- A1 and A2 share one PhoBERT + ResNet50 instance (saves ~600 MB of VRAM).
- B1 and B2 share one Qwen2-VL-2B-Instruct base; B1 runs with the LoRA adapter
  disabled, B2 with it enabled (saves ~3 GB).

Designed to run inside a Kaggle notebook with internet on. Launch with::

    uv run python app/demo.py --eager

Other flags::

    --a1-checkpoint reports/A1/checkpoints/best.pt
    --a2-checkpoint reports/A2/checkpoints/best.pt
    --b2-adapter    reports/B2/adapter
    --device        cuda|cpu
    --share         (only useful when not behind cloudflared)
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
# LawMate idiom: let CUDA allocate in non-contiguous chunks. With four models
# in VRAM, large activation tensors fragment the allocator and cause OOM even
# when total free VRAM looks adequate.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TRACK_INFO: dict[str, dict[str, str]] = {
    "A1": {
        "title": "A1 — Modular + LSTM",
        "arch": "PhoBERT-base + ResNet50 → CrossAttention → LSTM decoder",
        "note": "From-scratch architecture, frozen pretrained encoders, vocab=453.",
    },
    "A2": {
        "title": "A2 — Modular + Transformer",
        "arch": "PhoBERT-base + ResNet50 → CrossAttention → 2-layer Transformer decoder",
        "note": "Same encoders as A1; attention-based decoder over fused sequence.",
    },
    "B1": {
        "title": "B1 — Qwen2-VL zero-shot",
        "arch": "Qwen2-VL-2B-Instruct (NF4 4-bit), strict Vietnamese system prompt",
        "note": "Off-the-shelf multimodal LLM, no fine-tuning.",
    },
    "B2": {
        "title": "B2 — Qwen2-VL + LoRA SFT",
        "arch": "Qwen2-VL-2B-Instruct + LoRA r=16 on q/k/v/o, 3 epochs SFT",
        "note": "Same base as B1, fine-tuned on the FoodLensVN train split.",
    },
}


class _ModularRuntime:
    """Hold one PhoBERT + ResNet50 pair, shared between A1 and A2.

    Loading both modular tracks naively would put two PhoBERT-base copies
    (~500 MB each) and two ResNet50 copies in VRAM. They're frozen pretrained,
    so a single shared instance is correct and saves ~600 MB.
    """

    def __init__(self, ref_cfg_path: Path, device_str: str):
        import torch

        from src.data_loader.answer_tokenizer import AnswerTokenizer
        from src.models.encoders.image_encoder import (
            TimmImageEncoder,
            build_image_transform,
        )
        from src.models.encoders.text_encoder import (
            PhoBERTTextEncoder,
            load_phobert_tokenizer,
        )
        import scripts.train as train_mod

        cfg = train_mod._load_config(ref_cfg_path)
        md = cfg["model"]
        self.device = torch.device(device_str)

        self.image_encoder = (
            TimmImageEncoder(
                model_id=md["image_encoder"]["model_id"],
                pretrained=md["image_encoder"].get("pretrained", True),
                freeze=True,
            )
            .to(self.device)
            .eval()
        )
        self.text_encoder = (
            PhoBERTTextEncoder(model_id=md["text_encoder"]["model_id"], freeze=True)
            .to(self.device)
            .eval()
        )
        self.eval_tf = build_image_transform(self.image_encoder, is_training=False)
        self.phobert_tokenizer = load_phobert_tokenizer(md["text_encoder"]["model_id"])
        self.answer_tokenizer = AnswerTokenizer.load(cfg["dataset"]["answer_vocab"])
        self.max_q = cfg["loader"].get("max_question_length", 64)

    def build_predictor(
        self, cfg_path: Path, ckpt_path: Path
    ) -> Callable[[Any, str], str]:
        import torch

        from src.models.encoders.text_encoder import encode_questions
        from src.models.fusion.cross_attention import build_fusion
        from src.models.modular_vqa import ModularVQA
        import scripts.train as train_mod

        cfg = train_mod._load_config(cfg_path)
        md = cfg["model"]
        fusion = build_fusion(
            fusion_type=md["fusion"]["type"],
            d_img=self.image_encoder.feature_dim,
            d_txt=self.text_encoder.hidden_size,
            d_fused=md["fusion"]["d_fused"],
            n_heads=md["fusion"].get("n_heads", 8),
            dropout=md["fusion"].get("dropout", 0.1),
        )
        decoder = train_mod._build_decoder(
            md["decoder"],
            vocab_size=self.answer_tokenizer.vocab_size,
            d_fused=md["fusion"]["d_fused"],
        )
        model = (
            ModularVQA(
                image_encoder=self.image_encoder,
                text_encoder=self.text_encoder,
                fusion=fusion,
                decoder=decoder,
                decoder_input=md["decoder_input"],
            )
            .to(self.device)
            .eval()
        )

        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = state["model"] if "model" in state else state
        # Filter out encoder keys: encoders are shared and frozen pretrained,
        # so the checkpoint's encoder weights match the live ones exactly.
        # Loading them would just overwrite shared params with identical values
        # (and break sharing if for some reason they differ).
        sd_filtered = {
            k: v
            for k, v in sd.items()
            if not k.startswith(("image_encoder.", "text_encoder."))
        }
        model.load_state_dict(sd_filtered, strict=False)

        def predict(image: Any, question: str) -> str:
            pixel_values = self.eval_tf(image.convert("RGB")).unsqueeze(0).to(self.device)
            tok = encode_questions(
                self.phobert_tokenizer, [question], max_length=self.max_q
            )
            with torch.no_grad():
                ids = model.generate(
                    pixel_values,
                    tok["input_ids"].to(self.device),
                    tok["attention_mask"].to(self.device),
                )
            return self.answer_tokenizer.decode(ids[0].tolist())

        return predict


class _QwenRuntime:
    """Shared Qwen2-VL base for B1 and B2 (B1 runs with adapter disabled)."""

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
    """One column in the demo: holds a builder + cached predictor."""

    def __init__(
        self,
        name: str,
        builder: Callable[[], Callable[[Any, str], str]] | None,
    ):
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

    # Shared modular runtime: PhoBERT + ResNet50 loaded once, used by A1 and A2.
    a1_ckpt = Path(args.a1_checkpoint)
    a2_ckpt = Path(args.a2_checkpoint)
    modular_holder: dict[str, _ModularRuntime] = {}

    def get_modular() -> _ModularRuntime:
        if "rt" not in modular_holder:
            # Use whichever modular config is available as the reference for
            # encoder ids — A1 and A2 share the same encoder model_ids.
            ref_cfg = cfg_dir / ("A1.yaml" if a1_ckpt.exists() else "A2.yaml")
            modular_holder["rt"] = _ModularRuntime(ref_cfg, device_str)
        return modular_holder["rt"]

    def modular_builder(cfg_name: str, ckpt: Path) -> Callable[[], Callable[[Any, str], str]]:
        return lambda: get_modular().build_predictor(cfg_dir / cfg_name, ckpt)

    # Shared Qwen runtime for B1 + B2.
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
        return get_qwen().predict_b1

    def b2_builder() -> Callable[[Any, str], str]:
        return get_qwen().predict_b2

    tracks: list[_LazyTrack] = []
    tracks.append(
        _LazyTrack("A1", modular_builder("A1.yaml", a1_ckpt) if a1_ckpt.exists() else None)
    )
    tracks.append(
        _LazyTrack("A2", modular_builder("A2.yaml", a2_ckpt) if a2_ckpt.exists() else None)
    )
    tracks.append(_LazyTrack("B1", b1_builder))
    tracks.append(
        _LazyTrack("B2", b2_builder if b2_adapter.exists() else None)
    )
    return tracks


def _format_latency(secs: float) -> str:
    return f"⏱  {secs * 1000:.0f} ms" if secs < 1 else f"⏱  {secs:.2f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--a1-checkpoint", default="reports/A1/checkpoints/best.pt")
    parser.add_argument("--a2-checkpoint", default="reports/A2/checkpoints/best.pt")
    parser.add_argument("--b2-adapter", default="reports/B2/adapter")
    parser.add_argument("--device", default=None)
    parser.add_argument("--share", action="store_true", help="enable gradio share link")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="warm-load every track before launching gradio",
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
        for t in tracks:
            if not t.available:
                continue
            print(f"[eager] loading {t.name} ...", flush=True)
            t._predictor = t._builder()  # type: ignore[misc]
        print("[eager] all tracks warm; gradio is about to launch.", flush=True)

    import gradio as gr

    def run(image: Any, question: str):
        question = (question or "").strip()
        if image is None or not question:
            placeholder = "(provide image + question, then press Predict)"
            return [placeholder, "", placeholder, "", placeholder, "", placeholder, ""]

        outputs: list[str] = []
        for t in tracks:
            try:
                ans, secs = t.predict(image, question)
                outputs.append(ans)
                outputs.append(_format_latency(secs))
            except Exception as e:  # noqa: BLE001 — show the error in the UI rather than crashing
                outputs.append(f"[error] {type(e).__name__}: {e}")
                outputs.append("")
            # Free activation tensors between tracks so the next model's
            # forward pass doesn't trip over fragmented memory.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return outputs

    css = """
    .track-card { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 12px; }
    .track-title { font-weight: 600; font-size: 1.05em; margin-bottom: 4px; }
    .track-arch { color: var(--body-text-color-subdued); font-size: 0.85em; margin-bottom: 2px; }
    .track-note { color: var(--body-text-color-subdued); font-size: 0.8em; font-style: italic; margin-bottom: 8px; }
    .track-skip { color: #c0392b; font-size: 0.85em; font-style: italic; }
    """

    def card(track: _LazyTrack):
        info = TRACK_INFO[track.name]
        with gr.Group(elem_classes="track-card"):
            gr.Markdown(
                f"<div class='track-title'>{info['title']}</div>"
                f"<div class='track-arch'>{info['arch']}</div>"
                f"<div class='track-note'>{info['note']}</div>"
                + (
                    "<div class='track-skip'>checkpoint missing — track skipped</div>"
                    if not track.available
                    else ""
                )
            )
            answer = gr.Textbox(label="Answer", lines=2, interactive=False)
            latency = gr.Markdown("")
        return answer, latency

    with gr.Blocks(title="FoodLensVN — VQA Demo") as demo:
        gr.Markdown(
            "# FoodLensVN — Visual Question Answering on Vietnamese dishes\n"
            "Four configs run on the same input. **A1 / A2** are from-scratch modular models; "
            "**B1 / B2** use Qwen2-VL-2B-Instruct (B1 zero-shot, B2 with our LoRA adapter). "
            "Models are pre-loaded — first prediction is fast.",
            elem_id="demo-header",
        )

        outputs: list[gr.Component] = []
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                image_in = gr.Image(type="pil", label="Image of a Vietnamese dish")
                question_in = gr.Textbox(
                    label="Câu hỏi (tiếng Việt)",
                    placeholder="Đây là món gì?",
                    lines=2,
                )
                submit = gr.Button("Predict on all four", variant="primary")
                gr.Markdown(
                    "_Sample questions: « Đây là món gì? », « Món này có thịt không? », "
                    "« Có bao nhiêu chiếc nem? »_"
                )

            with gr.Column(scale=2):
                with gr.Row():
                    a1_ans, a1_lat = card(tracks[0])
                    a2_ans, a2_lat = card(tracks[1])
                with gr.Row():
                    b1_ans, b1_lat = card(tracks[2])
                    b2_ans, b2_lat = card(tracks[3])
                outputs = [a1_ans, a1_lat, a2_ans, a2_lat, b1_ans, b1_lat, b2_ans, b2_lat]

        submit.click(run, inputs=[image_in, question_in], outputs=outputs)
        question_in.submit(run, inputs=[image_in, question_in], outputs=outputs)

    demo.queue().launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.server_port,
        css=css,
    )


if __name__ == "__main__":
    main()
