from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_name_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            lines.append(s)
    return lines


class CharNameDataset(Dataset):
    def __init__(self, encoded: list[list[int]]):
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ids = self.encoded[idx]
        return torch.tensor(ids, dtype=torch.long), len(ids)


def collate_pad(pad_id: int):
    def _collate(batch: list[tuple[torch.Tensor, int]]):
        tensors, lens = zip(*batch)
        max_l = max(lens)
        B = len(batch)
        x = torch.full((B, max_l), pad_id, dtype=torch.long)
        lengths = torch.tensor(lens, dtype=torch.long)
        for i, t in enumerate(tensors):
            x[i, : t.numel()] = t
        return x, lengths

    return _collate
