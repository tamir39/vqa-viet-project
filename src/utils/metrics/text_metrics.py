"""Text-generation metrics: BLEU, ROUGE-L, METEOR.

All implementations use whitespace tokenization on already-normalized strings —
:func:`src.utils.vn_text.normalize_answer` should run upstream so casing,
punctuation, and number-words are canonical before scoring.
"""

from __future__ import annotations

from typing import Iterable


def _ensure_nltk_data() -> None:
    """Download wordnet/omw-1.4 lazily; METEOR needs them."""
    import nltk

    for pkg, path in [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def compute_bleu(preds: Iterable[str], golds: Iterable[str]) -> float:
    """Corpus BLEU-4 via NLTK (smoothing method 1, single reference each)."""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    refs = [[g.split()] for g in golds]
    hyps = [p.split() for p in preds]
    if not hyps:
        return 0.0
    return float(corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1))


def compute_rouge_l(preds: Iterable[str], golds: Iterable[str]) -> float:
    """Mean ROUGE-L F1 via ``rouge-score`` (no stemmer — Vietnamese)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    preds, golds = list(preds), list(golds)
    if not preds:
        return 0.0
    total = sum(scorer.score(g, p)["rougeL"].fmeasure for p, g in zip(preds, golds))
    return total / len(preds)


def compute_meteor(preds: Iterable[str], golds: Iterable[str]) -> float:
    """Mean METEOR via NLTK (single reference each)."""
    from nltk.translate.meteor_score import meteor_score

    _ensure_nltk_data()
    preds, golds = list(preds), list(golds)
    if not preds:
        return 0.0
    total = sum(
        meteor_score([g.split()], p.split()) for p, g in zip(preds, golds)
    )
    return total / len(preds)
