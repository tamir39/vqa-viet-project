"""Aggregate metric runner with per-type / per-difficulty breakdowns.

Designed to be called once per eval run from :mod:`scripts.eval`. Inputs are
already-normalized strings (run :func:`normalize_answer` upstream once); we do
not re-normalize here.

Returns a dict shaped like:

    {
      "overall":   {exact, soft, bleu, rouge_l, meteor, bertscore_f1, llm_judge?, n},
      "by_type":   {<type>: {...}, ...},
      "by_difficulty": {<level>: {...}, ...},
    }

Heavy metrics (bertscore, llm_judge) are computed once on the full set and
omitted from breakdowns to keep the eval fast.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from src.utils.metrics.text_metrics import compute_bleu, compute_meteor, compute_rouge_l
from src.utils.metrics.vqa_accuracy import compute_vqa_accuracy


def _slice(items: list[Any], idx: list[int]) -> list[Any]:
    return [items[i] for i in idx]


def _light_metrics(
    preds: list[str], golds: list[str], types: list[str]
) -> dict[str, float]:
    """VQA accuracy + BLEU/ROUGE-L/METEOR. No BERTScore, no LLM judge."""
    acc = compute_vqa_accuracy(preds, golds, types)
    return {
        **acc,
        "bleu": compute_bleu(preds, golds),
        "rouge_l": compute_rouge_l(preds, golds),
        "meteor": compute_meteor(preds, golds),
    }


def aggregate_metrics(
    preds: Iterable[str],
    golds: Iterable[str],
    types: Iterable[str],
    difficulties: Iterable[str],
    *,
    use_bertscore: bool = True,
    use_llm_judge: bool = False,
) -> dict[str, Any]:
    preds, golds = list(preds), list(golds)
    types, difficulties = list(types), list(difficulties)
    n = len(preds)
    if n == 0:
        return {"overall": {"n": 0}, "by_type": {}, "by_difficulty": {}}

    overall = _light_metrics(preds, golds, types)

    if use_bertscore:
        from src.utils.metrics.bertscore import compute_bertscore
        overall["bertscore_f1"] = compute_bertscore(preds, golds)

    if use_llm_judge:
        from src.utils.metrics.llm_judge import compute_llm_judge
        score = compute_llm_judge(preds, golds)
        if score >= 0.0:
            overall["llm_judge"] = score

    by_type: dict[str, dict[str, float]] = {}
    type_idx: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(types):
        type_idx[t].append(i)
    for t, idx in type_idx.items():
        by_type[t] = _light_metrics(
            _slice(preds, idx), _slice(golds, idx), _slice(types, idx)
        )

    by_diff: dict[str, dict[str, float]] = {}
    diff_idx: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(difficulties):
        diff_idx[d].append(i)
    for d, idx in diff_idx.items():
        by_diff[d] = _light_metrics(
            _slice(preds, idx), _slice(golds, idx), _slice(types, idx)
        )

    return {"overall": overall, "by_type": by_type, "by_difficulty": by_diff}
