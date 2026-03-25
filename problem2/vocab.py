from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharVocab:
    itos: list[str]
    stoi: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def build_vocab(
    names: list[str],
    start: str = "^",
    end: str = "$",
    pad: str = "<pad>",
) -> CharVocab:
    chars = {start, end, pad}
    for n in names:
        for c in n.strip().lower():
            chars.add(c)
    special = {start, end, pad}
    itos = [start, end, pad] + sorted(c for c in chars if c not in special)
    stoi = {c: i for i, c in enumerate(itos)}
    return CharVocab(itos=itos, stoi=stoi)


def wrap_name(name: str, start: str = "^", end: str = "$") -> str:
    return f"{start}{name.strip().lower()}{end}"
