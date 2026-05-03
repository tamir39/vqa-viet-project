"""Compact answer-side vocabulary tokenizer for the modular VQA decoder.

PhoBERT remains the question-side encoder; the decoder operates over a small,
domain-specific answer vocabulary built from canonicalized training answers.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.utils.vn_text import normalize_answer


class AnswerTokenizer:
    PAD: str = "<pad>"
    BOS: str = "<bos>"
    EOS: str = "<eos>"
    UNK: str = "<unk>"
    SPECIAL: tuple[str, ...] = (PAD, BOS, EOS, UNK)

    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    UNK_ID: int = 3

    MAX_LEN: int = 12

    def __init__(
        self,
        vocab: list[str],
        min_freq: int = 1,
        max_vocab_size: int = 1000,
    ) -> None:
        if list(vocab[: len(self.SPECIAL)]) != list(self.SPECIAL):
            raise ValueError(
                f"vocab must start with specials {self.SPECIAL}, "
                f"got {vocab[: len(self.SPECIAL)]}"
            )
        if min_freq < 1:
            raise ValueError(f"min_freq must be >= 1, got {min_freq}")
        if max_vocab_size <= len(self.SPECIAL):
            raise ValueError(
                f"max_vocab_size must be > {len(self.SPECIAL)} (number of specials), "
                f"got {max_vocab_size}"
            )

        self.vocab: list[str] = list(vocab)
        self.min_freq: int = int(min_freq)
        self.max_vocab_size: int = int(max_vocab_size)
        self.token2id: dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def build_from_corpus(
        cls,
        answers: Iterable[str],
        min_freq: int = 1,
        max_vocab_size: int = 1000,
    ) -> "AnswerTokenizer":
        if max_vocab_size <= len(cls.SPECIAL):
            raise ValueError(
                f"max_vocab_size must be > {len(cls.SPECIAL)}, got {max_vocab_size}"
            )

        counter: Counter[str] = Counter()
        for raw in answers:
            normalized = normalize_answer(raw)
            if normalized:
                counter.update(normalized.split())

        items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))

        budget = max_vocab_size - len(cls.SPECIAL)
        items = items[:budget]

        vocab = list(cls.SPECIAL) + [tok for tok, _ in items]
        return cls(vocab=vocab, min_freq=min_freq, max_vocab_size=max_vocab_size)

    def encode(self, text: str) -> list[int]:
        normalized = normalize_answer(text)
        tokens = normalized.split() if normalized else []

        content_budget = self.MAX_LEN - 2
        if len(tokens) > content_budget:
            tokens = tokens[:content_budget]

        ids: list[int] = [self.BOS_ID]
        for tok in tokens:
            ids.append(self.token2id.get(tok, self.UNK_ID))
        ids.append(self.EOS_ID)

        while len(ids) < self.MAX_LEN:
            ids.append(self.PAD_ID)

        return ids

    def decode(self, ids: Iterable[int]) -> str:
        specials = {self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID}
        out: list[str] = []
        for i in ids:
            if i in specials:
                continue
            if 0 <= i < len(self.vocab):
                out.append(self.vocab[i])
        return " ".join(out)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab": self.vocab,
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "AnswerTokenizer":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            vocab=list(data["vocab"]),
            min_freq=int(data.get("min_freq", 1)),
            max_vocab_size=int(data.get("max_vocab_size", 1000)),
        )
