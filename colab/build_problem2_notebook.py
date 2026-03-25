"""Emit colab/Problem2_Training.ipynb from problem2/*.py sources."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def code_cell(lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": ""},
        "outputs": [],
        "source": [lines] if not lines.endswith("\n") else [lines[:-1] + "\n"],
    }


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {"id": ""}, "source": [text]}

intro = r"""# Problem 2: Character-level Indian name generation

1. **Upload** `TrainingNames.txt` (1000 names) to the Colab root (`/content/`) via the file sidebar, *or* run the optional upload cell below.
2. **Run all cells** (GPU runtime recommended).
3. Download the folder `problem2_colab_out/` (checkpoints, `metrics_summary.json`, `samples_*.txt`) for your report.

Default hyperparameters: `embed_dim=64`, `hidden_dim=128`, `num_layers=1`, `lr=0.002`, `batch_size=32`, `epochs=40` (edit the hyperparameter cell to match your write-up).

After training, merge printed parameter counts and metrics into `output/PROBLEM2_REPORT.md`.
"""

upload_hint = r"""### Optional: upload from your laptop
```python
from google.colab import files
uploaded = files.upload()  # pick TrainingNames.txt; saves under /content/
```
"""

pip = """# Colab usually has torch. Install if ImportError.
try:
    import torch
except ImportError:
    %pip install -q torch
"""

setup = src("problem2/vocab.py") + "\n\n" + src("problem2/data_io.py")

models = src("problem2/models.py")

train_eval = """
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def build_model(kind, vocab_size, embed_dim, hidden_dim, num_layers):
    if kind == "rnn":
        return CharRNN(vocab_size, embed_dim, hidden_dim, num_layers=num_layers)
    if kind == "blstm":
        return PrefixBLSTM(vocab_size, embed_dim, hidden_dim, num_layers=num_layers)
    if kind == "attn":
        return RNNAttention(vocab_size, embed_dim, hidden_dim, num_layers=1)
    raise ValueError(kind)


def train_model(kind, names, out_dir, device, epochs=40, batch_size=32, lr=0.002, embed_dim=64, hidden_dim=128, num_layers=1):
    vocab = build_vocab(names)
    pad_id = vocab.stoi["<pad>"]
    encoded = [vocab.encode(wrap_name(n)) for n in names]
    ds = CharNameDataset(encoded)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad(pad_id))

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
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"train_meta_{kind}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"=== {kind} trainable parameters: {meta['trainable_params']} ===")

    for ep in range(1, epochs + 1):
        model.train()
        total, steps = 0.0, 0
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
        print(f"{kind} epoch {ep}/{epochs} loss={total/max(steps,1):.4f}")

    ckpt = {
        "model_kind": kind,
        "state_dict": model.state_dict(),
        "itos": vocab.itos,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    torch.save(ckpt, out_dir / f"checkpoint_{kind}.pt")
    return model, vocab, meta


def normalize_name(s):
    return " ".join(s.strip().lower().split())


@torch.no_grad()
def generate_many(model, vocab, n=400, max_len=45, temperature=0.85, device=None):
    device = device or next(model.parameters()).device
    start_id = vocab.stoi["^"]
    end_id = vocab.stoi["$"]
    out = []
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


def novelty_diversity(generated, train_set):
    if not generated:
        return 0.0, 0.0
    novel = sum(1 for g in generated if g not in train_set)
    uniq = len(set(generated))
    return novel / len(generated), uniq / len(generated)


# --- run ---
DATA_PATH = Path("TrainingNames.txt")
OUT_DIR = Path("problem2_colab_out")

names = load_name_lines(DATA_PATH)
print(f"Loaded {len(names)} training names from {DATA_PATH.resolve()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_set = {normalize_name(x) for x in names}

results = {}
for kind in ("rnn", "blstm", "attn"):
    _, vocab, meta = train_model(
        kind,
        names,
        OUT_DIR,
        device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )
    ckpt_path = OUT_DIR / f"checkpoint_{kind}.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(kind, len(ckpt["itos"]), ckpt["embed_dim"], ckpt["hidden_dim"], ckpt.get("num_layers", 1))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    gen = generate_many(model, vocab, n=GEN_N, device=device)
    nov, div = novelty_diversity(gen, train_set)
    results[kind] = {"novelty_rate": nov, "diversity": div, "params": meta["trainable_params"]}
    (OUT_DIR / f"samples_{kind}.txt").write_text(chr(10).join(gen[:120]) + chr(10), encoding="utf-8")
    print(f"{kind}: novelty={nov:.3f} diversity={div:.3f}")

(OUT_DIR / "metrics_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("Saved checkpoints and metrics under", OUT_DIR.resolve())
"""

main_params = """# Hyperparameters (TASK-1)
EPOCHS = 40
BATCH_SIZE = 32
LR = 0.002
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1
GEN_N = 500
"""

cells = [
    md_cell(intro),
    md_cell(upload_hint),
    code_cell(pip),
    code_cell(main_params + "\n" + setup),
    code_cell(models),
    code_cell(train_eval),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out_ipynb = ROOT / "colab" / "Problem2_Training.ipynb"
out_ipynb.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", out_ipynb)
