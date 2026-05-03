"""LSTM decoder for the modular VQA pipeline (A1).

Operates over the compact :class:`AnswerTokenizer` vocabulary (PAD/BOS/EOS/UNK
at fixed ids 0–3, ``MAX_LEN=12``). The fused image-question vector from the
fusion module is projected into the LSTM's initial hidden + cell state; token
generation is greedy.

Forward signature
    forward(fused_pooled: (B, D_fused), target_ids: (B, T)) -> logits (B, T, V)
Generation
    generate(fused_pooled) -> ids (B, <=MAX_LEN), starting with BOS, padded after EOS.
"""

from __future__ import annotations

import torch
from torch import nn

from src.data_loader.answer_tokenizer import AnswerTokenizer


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_fused: int,
        d_embed: int = 256,
        d_hidden: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_len: int = AnswerTokenizer.MAX_LEN,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.max_len = max_len

        self.embed = nn.Embedding(
            vocab_size, d_embed, padding_idx=AnswerTokenizer.PAD_ID
        )
        self.init_h = nn.Linear(d_fused, num_layers * d_hidden)
        self.init_c = nn.Linear(d_fused, num_layers * d_hidden)
        self.lstm = nn.LSTM(
            input_size=d_embed,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(d_hidden, vocab_size)

    def _init_state(
        self, fused_pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = fused_pooled.size(0)
        h = (
            self.init_h(fused_pooled)
            .view(B, self.num_layers, self.d_hidden)
            .transpose(0, 1)
            .contiguous()
        )
        c = (
            self.init_c(fused_pooled)
            .view(B, self.num_layers, self.d_hidden)
            .transpose(0, 1)
            .contiguous()
        )
        return h, c

    def forward(
        self, fused_pooled: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        emb = self.embed(target_ids)
        h0, c0 = self._init_state(fused_pooled)
        out, _ = self.lstm(emb, (h0, c0))
        return self.out_proj(out)

    @torch.no_grad()
    def generate(
        self,
        fused_pooled: torch.Tensor,
        max_len: int | None = None,
    ) -> torch.Tensor:
        max_len = max_len or self.max_len
        B = fused_pooled.size(0)
        device = fused_pooled.device

        h, c = self._init_state(fused_pooled)
        cur = torch.full(
            (B, 1), AnswerTokenizer.BOS_ID, dtype=torch.long, device=device
        )
        out_ids: list[torch.Tensor] = [cur]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            emb = self.embed(cur)
            step_out, (h, c) = self.lstm(emb, (h, c))
            logits = self.out_proj(step_out[:, -1])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            next_tok = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_tok, AnswerTokenizer.PAD_ID),
                next_tok,
            )
            out_ids.append(next_tok)
            finished = finished | (next_tok.squeeze(-1) == AnswerTokenizer.EOS_ID)
            if finished.all():
                break
            cur = next_tok

        return torch.cat(out_ids, dim=1)
