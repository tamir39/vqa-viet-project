"""Vietnamese answer canonicalization for FoodLensVN VQA.

Single source of truth for normalizing model outputs and gold answers before
any comparison, scoring, or vocabulary building.
"""

from __future__ import annotations

import re

_YES_ALIASES: frozenset[str] = frozenset({"co", "có", "yes"})
_NO_ALIASES: frozenset[str] = frozenset({"khong", "không", "no"})

_NUMBER_MAP: dict[str, str] = {
    "khong": "0", "không": "0",
    "mot": "1", "một": "1",
    "hai": "2",
    "ba": "3",
    "bon": "4", "bốn": "4", "tu": "4", "tư": "4",
    "nam": "5", "năm": "5",
    "sau": "6", "sáu": "6",
    "bay": "7", "bảy": "7",
    "tam": "8", "tám": "8",
    "chin": "9", "chín": "9",
    "muoi": "10", "mười": "10",
}

_CLASSIFIERS: frozenset[str] = frozenset({"cái", "miếng", "phần", "tô", "bát"})

_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.,!?…:;]+$")


def normalize_answer(text: str) -> str:
    """Canonicalize a Vietnamese VQA answer to a stable comparable form.

    Pipeline (whole-string yes/no match short-circuits before number mapping
    so a standalone "không" stays as the yes/no token rather than the digit "0"):

      1. lowercase, trim, collapse internal whitespace
      2. strip trailing punctuation
      3. yes/no whole-string match -> "có" / "không" (return)
      4. per-token Vietnamese number-word -> digit
      5. strip trailing classifier words
    """
    if text is None:
        return ""

    s = _WHITESPACE_RE.sub(" ", text.lower().strip())
    s = _TRAILING_PUNCT_RE.sub("", s).strip()

    if not s:
        return ""

    if s in _YES_ALIASES:
        return "có"
    if s in _NO_ALIASES:
        return "không"

    tokens = [_NUMBER_MAP.get(tok, tok) for tok in s.split()]
    while tokens and tokens[-1] in _CLASSIFIERS:
        tokens.pop()

    return " ".join(tokens).strip()
