"""LLM-as-judge: prompts a local LLM for binary equivalence (yes/no).

Default judge is the same Qwen2-VL we already load for B1/B2 (used here in
text-only mode), so no extra model needs to ship for offline runs. The judge
reads ``LLM_JUDGE_DISABLE=1`` as an opt-out — useful for fast smoke runs and
CI where loading a 2B model is overkill.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable


JUDGE_SYSTEM = (
    "Bạn là một trọng tài đánh giá VQA tiếng Việt. "
    "Nhiệm vụ của bạn là quyết định hai câu trả lời sau có cùng nghĩa hay không. "
    "Chỉ trả lời 'có' hoặc 'không', không giải thích."
)
_YES_RE = re.compile(r"\bcó\b", re.IGNORECASE)


def _ask(model: Any, processor: Any, pred: str, gold: str) -> bool:
    import torch

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Câu 1: {gold}\nCâu 2: {pred}\nCó cùng nghĩa không?"},
            ],
        },
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text_prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    decoded = processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]
    return bool(_YES_RE.search(decoded))


def compute_llm_judge(
    preds: Iterable[str],
    golds: Iterable[str],
    model: Any | None = None,
    processor: Any | None = None,
) -> float:
    """Fraction of (pred, gold) pairs the judge calls equivalent.

    If the judge is disabled by env var, returns ``-1.0`` to signal "skipped"
    (the aggregate layer drops keys with negative values).
    """
    if os.environ.get("LLM_JUDGE_DISABLE") == "1":
        return -1.0

    preds, golds = list(preds), list(golds)
    if not preds:
        return 0.0

    if model is None or processor is None:
        from src.models.multimodal.qwen_vl import load_qwen_vl

        model, processor = load_qwen_vl(quantize_4bit=True)

    hits = sum(1 for p, g in zip(preds, golds) if _ask(model, processor, p, g))
    return hits / len(preds)
