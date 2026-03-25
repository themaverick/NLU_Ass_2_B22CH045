"""Train character-level name models (RNN, Prefix BLSTM, RNN+Attention)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from problem2.data_io import CharNameDataset, collate_pad, load_name_lines
from problem2.models import CharRNN, PrefixBLSTM, RNNAttention, count_trainable_params
from problem2.vocab import build_vocab, wrap_name


def build_model(kind: str, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int):
    if kind == "rnn":
        return CharRNN(vocab_size, embed_dim, hidden_dim, num_layers=num_layers)
    if kind == "blstm":
        return PrefixBLSTM(vocab_size, embed_dim, hidden_dim, num_layers=num_layers)
    if kind == "attn":
        return RNNAttention(vocab_size, embed_dim, hidden_dim, num_layers=1)
    raise ValueError(f"Unknown model kind {kind!r}")


def train(
    data_path: Path,
    out_dir: Path,
    kind: str,
    epochs: int,
    batch_size: int,
    lr: float,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
) -> None:
    names = load_name_lines(data_path)
    if len(names) < 10:
        raise SystemExit(f"Too few names in {data_path}")

    vocab = build_vocab(names)
    pad_id = vocab.stoi["<pad>"]
    start_id = vocab.stoi["^"]
    end_id = vocab.stoi["$"]

    encoded = []
    for n in names:
        s = wrap_name(n)
        encoded.append(vocab.encode(s))

    ds = CharNameDataset(encoded)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pad(pad_id),
    )

    model = build_model(kind, vocab.size, embed_dim, hidden_dim, num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    meta = {
        "model": kind,
        "trainable_params": count_trainable_params(model),
        "vocab_size": vocab.size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_train_names": len(names),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for x, lengths in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            opt.zero_grad()
            if isinstance(model, CharRNN):
                loss = model.loss(x, pad_idx=pad_id)
            else:
                loss = model.batched_loss(x, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
            steps += 1
        avg = total / max(steps, 1)
        print(f"epoch {ep}/{epochs}  loss={avg:.4f}")

    ckpt = {
        "model_kind": kind,
        "state_dict": model.state_dict(),
        "itos": vocab.itos,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    torch.save(ckpt, out_dir / f"checkpoint_{kind}.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output/problem2"))
    ap.add_argument("--model", choices=("rnn", "blstm", "attn"), required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(
        args.data,
        args.out,
        args.model,
        args.epochs,
        args.batch_size,
        args.lr,
        args.embed_dim,
        args.hidden_dim,
        args.num_layers,
        device,
    )


if __name__ == "__main__":
    main()
