#!/usr/bin/env python3
"""
Problem 2 — train RNN, Prefix BLSTM, and RNN+Attention on GPU (Colab-friendly).

Run from the **repository root** (so `problem2` imports work):
  python -m problem2.colab_train_all --data TrainingNames.txt --out problem2_colab_out

On Google Colab (upload the whole `problem2` package + `TrainingNames.txt`):
  !pip install -q matplotlib
  import sys
  sys.path.insert(0, "/content")   # if you uploaded repo as /content/<repo_name>
  # then: !python /content/<repo_name>/problem2/colab_train_all.py --data /content/TrainingNames.txt --out /content/problem2_colab_out

Writes per-model:
  - checkpoint_<kind>.pt
  - loss_history_<kind>.json
  - loss_curve_<kind>.png
  - train_meta_<kind>.json
Plus summary.json with all paths and hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Allow `python problem2/colab_train_all.py` from repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
    raise ValueError(kind)


def train_one_model(
    kind: str,
    names: list[str],
    out_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
) -> dict:
    vocab = build_vocab(names)
    pad_id = vocab.stoi["<pad>"]
    encoded = [vocab.encode(wrap_name(n)) for n in names]
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
        "model_kind": kind,
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

    loss_history: list[float] = []

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
        loss_history.append(avg)
        print(f"  [{kind}] epoch {ep}/{epochs}  loss={avg:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"train_meta_{kind}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (out_dir / f"loss_history_{kind}.json").write_text(
        json.dumps({"epochs": list(range(1, epochs + 1)), "loss": loss_history}, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epochs + 1), loss_history, marker=".", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean batch loss")
    ax.set_title(f"Training loss — {kind.upper()}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_png = out_dir / f"loss_curve_{kind}.png"
    fig.savefig(loss_png, dpi=150)
    plt.close(fig)
    print(f"  saved {loss_png}")

    ckpt = {
        "model_kind": kind,
        "state_dict": model.state_dict(),
        "itos": vocab.itos,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    ckpt_path = out_dir / f"checkpoint_{kind}.pt"
    torch.save(ckpt, ckpt_path)
    print(f"  saved {ckpt_path}")

    return {
        "kind": kind,
        "meta": meta,
        "loss_history": loss_history,
        "checkpoint": str(ckpt_path),
        "loss_plot": str(loss_png),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train all Problem 2 char-level models (Colab GPU).")
    ap.add_argument("--data", type=Path, default=Path("TrainingNames.txt"), help="Path to TrainingNames.txt")
    ap.add_argument("--out", type=Path, default=Path("problem2_colab_out"), help="Output directory")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--models", default="rnn,blstm,attn", help="Comma-separated: rnn,blstm,attn")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("(Warning: training on CPU is slow; use Colab GPU for full runs.)")

    names = load_name_lines(args.data)
    if len(names) < 10:
        raise SystemExit(f"Too few names in {args.data}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    kinds = [k.strip() for k in args.models.split(",") if k.strip()]
    results: list[dict] = []

    for kind in kinds:
        print(f"\n=== Training {kind} ===")
        results.append(
            train_one_model(
                kind,
                names,
                out_dir,
                device,
                args.epochs,
                args.batch_size,
                args.lr,
                args.embed_dim,
                args.hidden_dim,
                args.num_layers,
            )
        )

    summary = {
        "data_path": str(args.data.resolve()),
        "out_dir": str(out_dir.resolve()),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
        },
        "runs": results,
    }
    summary_path = out_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    print("Download the whole output folder for analysis.")


if __name__ == "__main__":
    main()
