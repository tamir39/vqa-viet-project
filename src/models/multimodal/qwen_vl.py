"""Qwen2-VL-2B-Instruct loader, prompt builder, and inference helper for B1/B2.

Designed for Kaggle GPU runtime (<=16 GB VRAM): 4-bit NF4 quantization is the
default load path; bf16 is the fallback for non-quantized inference. Set
``KAGGLE_NO_INTERNET=1`` to force ``local_files_only=True`` on both model and
processor downloads.
"""

from __future__ import annotations

import os
import re
from typing import Any

from src.utils.vn_text import normalize_answer


SYSTEM: str = (
    "Bạn là hệ thống VQA. Trả lời bằng tiếng Việt. "
    "Chỉ trả lời, không giải thích. Tối đa 10 từ."
)

DEFAULT_MODEL_ID: str = "Qwen/Qwen2-VL-2B-Instruct"
MAX_NEW_TOKENS: int = 20

_PROMPT_ECHO_RE = re.compile(r"(?:câu hỏi|trả lời)\s*:\s*", flags=re.IGNORECASE)
_TRAILING_PUNCT_RE = re.compile(r"[.,!?…:;]+$")


def build_messages(image: Any, question: str) -> list[dict]:
    """Construct the chat-template message list with the strict VQA system prompt."""
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Câu hỏi: {question}\nTrả lời:"},
            ],
        },
    ]


def _clean_output(text: str) -> str:
    """Strip prompt echoes and punctuation, then canonicalize the model output."""
    if not text:
        return ""
    s = text.split("\n", 1)[0]
    s = _PROMPT_ECHO_RE.sub("", s)
    s = _TRAILING_PUNCT_RE.sub("", s.strip()).strip()
    return normalize_answer(s)


def load_qwen_vl(
    model_id: str = DEFAULT_MODEL_ID,
    quantize_4bit: bool = True,
) -> tuple[Any, Any]:
    """Load Qwen2-VL model + processor. NF4 4-bit by default; bf16 otherwise."""
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    local_files_only = os.environ.get("KAGGLE_NO_INTERNET") == "1"

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "local_files_only": local_files_only,
    }

    if quantize_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    return model, processor


def generate(
    model: Any,
    processor: Any,
    image: Any,
    question: str,
) -> str:
    """Run greedy generation for one (image, question) pair and return a clean answer."""
    import torch

    messages = build_messages(image, question)
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated = output_ids[:, prompt_len:]
    decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return _clean_output(decoded)
