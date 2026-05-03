"""Augmentation utilities for the modular VQA pipeline.

Two surfaces:

* :func:`build_image_train_transform` — a torchvision transform that wraps
  the timm-resolved eval transform with mild train-time augmentation
  (random resized crop, horizontal flip, color jitter, RandAugment). The
  base normalization (mean/std/resize) comes from
  :func:`src.models.encoders.image_encoder.build_image_transform` so the
  image distribution stays compatible with the backbone's pretraining.

* :func:`augment_question` — text-side paraphrase / synonym swap with a
  small Vietnamese synonym map. Back-translation is **explicitly excluded**
  (corrupts canonical answer forms; called out in `PLANNING.md` §9).
  Augmentation is applied at the dataset / collate boundary by the caller;
  this module only provides the function.
"""

from __future__ import annotations

import random
from typing import Any

from torchvision import transforms as T


VN_QUESTION_SYNONYMS: dict[str, list[str]] = {
    "món": ["món ăn", "đĩa"],
    "món này": ["đĩa này", "tô này"],
    "có": ["có phải"],
    "bao nhiêu": ["mấy"],
    "phía nào": ["bên nào"],
    "đây là": ["kia là"],
    "thường": [""],
    "không": ["không phải"],
}


def build_image_train_transform(
    eval_transform: Any,
    image_size: int = 224,
    color_jitter: float = 0.2,
    use_randaugment: bool = True,
) -> Any:
    """Wrap the timm eval transform with mild train-time augmentation.

    The eval transform already handles backbone-correct resize +
    normalization; we replace its deterministic resize with a
    ``RandomResizedCrop`` at the same resolution, and prepend horizontal
    flip + color jitter (+ optional RandAugment) before normalization.
    """
    aug_ops: list[Any] = [
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
        ),
    ]
    if use_randaugment:
        aug_ops.append(T.RandAugment(num_ops=2, magnitude=7))
    aug_ops.append(T.ToTensor())

    norm = _extract_normalize(eval_transform)
    if norm is not None:
        aug_ops.append(norm)

    return T.Compose(aug_ops)


def _extract_normalize(transform: Any) -> T.Normalize | None:
    """Pull the trailing :class:`torchvision.transforms.Normalize` from a Compose."""
    if isinstance(transform, T.Compose):
        for t in reversed(transform.transforms):
            if isinstance(t, T.Normalize):
                return t
    elif isinstance(transform, T.Normalize):
        return transform
    return None


def augment_question(question: str, p: float = 0.3, rng: random.Random | None = None) -> str:
    """Apply a single random synonym swap on the question with probability ``p``.

    Conservative by design: at most one phrase is replaced per call to keep
    the question semantically intact. Returns the question unchanged if no
    matching phrase is present or the random gate fails.
    """
    rng = rng or random
    if rng.random() >= p:
        return question

    candidates = [phrase for phrase in VN_QUESTION_SYNONYMS if phrase in question]
    if not candidates:
        return question

    phrase = rng.choice(candidates)
    replacement = rng.choice(VN_QUESTION_SYNONYMS[phrase])
    return question.replace(phrase, replacement, 1).strip()
