"""BERTScore wrapper using xlm-roberta-base.

Honors ``KAGGLE_NO_INTERNET=1`` by passing ``use_fast_tokenizer=True`` and
relying on the HF cache pre-staged via the offline workflow. Returns mean F1.
"""

from __future__ import annotations

import os
from typing import Iterable


DEFAULT_MODEL: str = "xlm-roberta-base"


def compute_bertscore(
    preds: Iterable[str],
    golds: Iterable[str],
    model_type: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> float:
    """Mean BERTScore F1 across the corpus."""
    from bert_score import score as bert_score_score

    preds, golds = list(preds), list(golds)
    if not preds:
        return 0.0

    # bert_score's sent_encode hits a removed tokenizer API on empty strings
    # (XLMRobertaTokenizer.build_inputs_with_special_tokens). Substitute a
    # single space — scoring an empty prediction against a non-empty gold
    # already yields ~0 similarity, so the metric stays meaningful.
    preds = [p if p.strip() else " " for p in preds]
    golds = [g if g.strip() else " " for g in golds]

    _, _, f1 = bert_score_score(
        cands=preds,
        refs=golds,
        model_type=model_type,
        lang="vi",
        batch_size=batch_size,
        verbose=False,
        rescale_with_baseline=False,
        use_fast_tokenizer=True,
    )
    return float(f1.mean().item())
