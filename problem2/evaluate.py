"""TASK-2: novelty rate and diversity for generated names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from problem2.data_io import load_name_lines
from problem2.models import CharRNN, PrefixBLSTM, RNNAttention
from problem2.vocab import CharVocab


def load_vocab_from_checkpoint(ckpt: dict) -> CharVocab:
    return CharVocab(itos=list(ckpt["itos"]), stoi={c: i for i, c in enumerate(ckpt["itos"])})


def load_model(kind: str, ckpt: dict, device: torch.device):
    vocab_size = len(ckpt["itos"])
    h = ckpt["hidden_dim"]
    e = ckpt["embed_dim"]
    nl = ckpt.get("num_layers", 1)
    if kind == "rnn":
        m = CharRNN(vocab_size, e, h, num_layers=nl)
    elif kind == "blstm":
        m = PrefixBLSTM(vocab_size, e, h, num_layers=nl)
    elif kind == "attn":
        m = RNNAttention(vocab_size, e, h, num_layers=1)
    else:
        raise ValueError(kind)
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).eval()


def normalize_name(s: str) -> str:
    return " ".join(s.strip().lower().split())


@torch.no_grad()
def generate_many(
    model,
    vocab: CharVocab,
    n: int,
    max_len: int,
    temperature: float,
    device: torch.device,
) -> list[str]:
    start_id = vocab.stoi["^"]
    end_id = vocab.stoi["$"]
    out: list[str] = []
    for _ in range(n):
        ids = model.generate(start_id, end_id, max_len=max_len, temperature=temperature, device=device)
        chars = []
        for i in ids[1:]:
            if i == end_id:
                break
            c = vocab.itos[i]
            if c in ("^", "$", "<pad>"):
                continue
            chars.append(c)
        name = normalize_name("".join(chars))
        if name:
            out.append(name)
    return out


def novelty_and_diversity(generated: list[str], train_set: set[str]) -> dict:
    if not generated:
        return {"novelty_rate": 0.0, "diversity": 0.0, "n": 0, "unique": 0}
    novel = sum(1 for g in generated if g not in train_set)
    uniq = len(set(generated))
    return {
        "novelty_rate": novel / len(generated),
        "diversity": uniq / len(generated),
        "n": len(generated),
        "unique": uniq,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--train-data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500, help="Number of samples to generate")
    ap.add_argument("--max-len", type=int, default=45)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    kind = ckpt["model_kind"]
    model = load_model(kind, ckpt, device)
    vocab = load_vocab_from_checkpoint(ckpt)

    train_names = {normalize_name(x) for x in load_name_lines(args.train_data)}
    gen = generate_many(model, vocab, args.n, args.max_len, args.temperature, device)
    metrics = novelty_and_diversity(gen, train_names)
    metrics["model"] = kind

    print(json.dumps(metrics, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (args.out_json.with_suffix(".samples.txt")).write_text("\n".join(gen[:200]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
