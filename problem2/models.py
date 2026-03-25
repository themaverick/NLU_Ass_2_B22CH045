"""
Character-level name generators (implemented with PyTorch).

1. CharRNN — vanilla RNN (tanh) language model over characters.
2. PrefixBLSTM — two LSTMs on the prefix only (forward + backward on reversed prefix);
   both directions see only past characters, so the model is a proper autoregressive LM.
3. RNNAttention — vanilla RNN with additive attention over past hidden states at each step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharRNN(nn.Module):
    """Single-layer (or stacked) tanh RNN; predicts next character from hidden state."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        """x: (B, T) token ids. Returns logits (B, T, V), h_n (num_layers, B, H)."""
        emb = self.embedding(x)
        out, h_n = self.rnn(emb, h0)
        logits = self.fc(out)
        return logits, h_n

    def loss(self, x: torch.Tensor, pad_idx: int | None = None) -> torch.Tensor:
        """Teacher forcing: predict x[:, 1:] from x[:, :-1]."""
        inp, tgt = x[:, :-1], x[:, 1:]
        logits, _ = self.forward(inp)
        if pad_idx is not None:
            mask = tgt != pad_idx
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="none",
            )
            return (ce * mask.reshape(-1).float()).sum() / mask.sum().clamp(min=1).float()
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
        )

    @torch.no_grad()
    def generate(
        self,
        start_id: int,
        end_id: int,
        max_len: int = 40,
        temperature: float = 0.9,
        device: torch.device | None = None,
    ) -> list[int]:
        device = device or next(self.parameters()).device
        ids = [start_id]
        h = torch.zeros(self.num_layers, 1, self.hidden_dim, device=device)
        for _ in range(max_len):
            x = torch.tensor([[ids[-1]]], device=device, dtype=torch.long)
            emb = self.embedding(x)
            out, h = self.rnn(emb, h)
            logits = self.fc(out[:, -1, :]) / temperature
            p = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(p, 1).item()
            ids.append(nxt)
            if nxt == end_id:
                break
        return ids


class PrefixBLSTM(nn.Module):
    """Prefix-only bidirectional LSTM: at step t uses forward LSTM on x[:t+1] and backward on rev(x[:t+1])."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_f = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_b = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward_prefix_logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, L) full sequence with start/end tokens. Returns logits for positions 0..L-2 -> predict x[1:]."""
        emb = self.embedding(x)
        _, L, _ = emb.shape
        logits_list: list[torch.Tensor] = []
        for t in range(L - 1):
            pref = emb[:, : t + 1, :]
            _, (hf, _) = self.lstm_f(pref)
            _, (hb, _) = self.lstm_b(torch.flip(pref, dims=[1]))
            h = torch.cat([hf[-1], hb[-1]], dim=-1)
            logits_list.append(self.fc(h))
        return torch.stack(logits_list, dim=1)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_prefix_logits(x)
        tgt = x[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
        )

    def batched_loss(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for b in range(x.size(0)):
            L = int(lengths[b].item())
            if L < 2:
                continue
            parts.append(self.loss(x[b : b + 1, :L]))
        if not parts:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        return torch.stack(parts).mean()

    @torch.no_grad()
    def generate(
        self,
        start_id: int,
        end_id: int,
        max_len: int = 40,
        temperature: float = 0.9,
        device: torch.device | None = None,
    ) -> list[int]:
        device = device or next(self.parameters()).device
        ids = [start_id]
        for _ in range(max_len):
            x = torch.tensor([ids], device=device, dtype=torch.long)
            emb = self.embedding(x)
            t = x.size(1) - 1
            pref = emb[:, : t + 1, :]
            _, (hf, _) = self.lstm_f(pref)
            _, (hb, _) = self.lstm_b(torch.flip(pref, dims=[1]))
            h = torch.cat([hf[-1], hb[-1]], dim=-1)
            logits = self.fc(h) / temperature
            p = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(p, 1).item()
            ids.append(nxt)
            if nxt == end_id:
                break
        return ids


class RNNAttention(nn.Module):
    """Single-layer RNN with Bahdanau-style attention over previous hidden states."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        if num_layers != 1:
            raise ValueError("RNNAttention implementation uses num_layers=1 for clarity.")
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, 1, batch_first=True, nonlinearity="tanh")
        self.attn_w = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward_prefix_logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, L)."""
        emb = self.embedding(x)
        _, L, _ = emb.shape
        out, _ = self.rnn(emb)
        logits_list: list[torch.Tensor] = []
        scale = 1.0 / math.sqrt(self.hidden_dim)
        for t in range(L - 1):
            h_t = out[:, t : t + 1, :]
            hist = out[:, : t + 1, :]
            h_t_rep = h_t.expand(-1, hist.size(1), -1)
            energy = self.attn_v(torch.tanh(self.attn_w(torch.cat([hist, h_t_rep], dim=-1))))
            attn = F.softmax(energy.squeeze(-1) * scale, dim=-1)
            ctx = torch.bmm(attn.unsqueeze(1), hist).squeeze(1)
            combined = torch.cat([h_t.squeeze(1), ctx], dim=-1)
            logits_list.append(self.fc(combined))
        return torch.stack(logits_list, dim=1)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_prefix_logits(x)
        tgt = x[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
        )

    def batched_loss(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for b in range(x.size(0)):
            L = int(lengths[b].item())
            if L < 2:
                continue
            parts.append(self.loss(x[b : b + 1, :L]))
        if not parts:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        return torch.stack(parts).mean()

    @torch.no_grad()
    def generate(
        self,
        start_id: int,
        end_id: int,
        max_len: int = 40,
        temperature: float = 0.9,
        device: torch.device | None = None,
    ) -> list[int]:
        device = device or next(self.parameters()).device
        ids = [start_id]
        emb_cache: list[torch.Tensor] = []
        h = torch.zeros(1, 1, self.hidden_dim, device=device)
        for _ in range(max_len):
            x = torch.tensor([[ids[-1]]], device=device, dtype=torch.long)
            e = self.embedding(x)
            o, h = self.rnn(e, h)
            emb_cache.append(o)
            hist = torch.cat(emb_cache, dim=1)
            t = hist.size(1) - 1
            h_t = hist[:, t : t + 1, :]
            energy = self.attn_v(
                torch.tanh(
                    self.attn_w(
                        torch.cat([hist, h_t.expand(-1, hist.size(1), -1)], dim=-1)
                    )
                )
            )
            attn = F.softmax(energy.squeeze(-1) / math.sqrt(self.hidden_dim), dim=-1)
            ctx = torch.bmm(attn.unsqueeze(1), hist).squeeze(1)
            combined = torch.cat([h_t.squeeze(1), ctx], dim=-1)
            logits = self.fc(combined) / temperature
            p = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(p, 1).item()
            ids.append(nxt)
            if nxt == end_id:
                break
        return ids


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
