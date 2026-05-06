"""Type-aware VQA accuracy.

* ``exact``       — string equality after :func:`normalize_answer`.
* ``soft``        — type-aware:
    - ``yes_no``    : exact-match on canonical {"có", "không"}.
    - ``counting``  : digit-equality (extract first integer from each side).
    - others        : token-overlap F1 (Squad-style).
"""

from __future__ import annotations

import re
from typing import Iterable

from src.utils.vn_text import normalize_answer


_INT_RE = re.compile(r"-?\d+")


def _first_int(s: str) -> str | None:
    m = _INT_RE.search(s)
    return m.group(0) if m else None


def _token_f1(pred: str, gold: str) -> float:
    pt = pred.split()
    gt = gold.split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    common: dict[str, int] = {}
    g_count: dict[str, int] = {}
    for t in gt:
        g_count[t] = g_count.get(t, 0) + 1
    overlap = 0
    for t in pt:
        if g_count.get(t, 0) > 0:
            overlap += 1
            g_count[t] -= 1
            common[t] = common.get(t, 0) + 1
    if overlap == 0:
        return 0.0
    p = overlap / len(pt)
    r = overlap / len(gt)
    return 2 * p * r / (p + r)


def _soft_match(pred: str, gold: str, qtype: str) -> float:
    if qtype == "yes_no":
        return 1.0 if pred == gold and pred in ("có", "không") else 0.0
    if qtype == "counting":
        pn, gn = _first_int(pred), _first_int(gold)
        if pn is None or gn is None:
            return 0.0
        return 1.0 if pn == gn else 0.0
    return _token_f1(pred, gold)


def compute_vqa_accuracy(
    preds: Iterable[str],
    golds: Iterable[str],
    types: Iterable[str],
) -> dict[str, float]:
    preds = [normalize_answer(p) for p in preds]
    golds = [normalize_answer(g) for g in golds]
    types = list(types)

    if not preds:
        return {"exact": 0.0, "soft": 0.0, "n": 0}

    exact = sum(1 for p, g in zip(preds, golds) if p == g) / len(preds)
    soft = sum(_soft_match(p, g, t) for p, g, t in zip(preds, golds, types)) / len(preds)
    return {"exact": exact, "soft": soft, "n": len(preds)}
