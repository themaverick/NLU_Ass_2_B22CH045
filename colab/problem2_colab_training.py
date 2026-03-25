#!/usr/bin/env python3
"""
Problem 2 — GPU training driver for Colab (or local).

Split this file into notebook cells as you like: install deps → optional upload /
Gemini → train → optional zip.

Examples
--------
Colab (defaults assume repo at /content/NLU_assignment_2 and names at /content):

  python colab/problem2_colab_training.py \\
    --repo /content/NLU_assignment_2 \\
    --data /content/TrainingNames.txt \\
    --out /content/problem2_colab_out \\
    --install-deps

Local (from repo root; repo inferred from this file if --repo omitted):

  python colab/problem2_colab_training.py --data TrainingNames.txt

Optional Colab-only helpers (uncomment bodies or call from a notebook cell):

  - upload_training_names_via_colab(destination)
  - generate_training_names_gemini_colab(repo, destination, ...)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _default_repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_data_path(repo: Path, data: Path | None) -> Path:
    if data is not None:
        return data
    for candidate in (repo / "TrainingNames.txt", repo / "problem2" / "data" / "TrainingNames.txt"):
        if candidate.is_file():
            return candidate
    return repo / "TrainingNames.txt"


def install_matplotlib_torch() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "matplotlib", "torch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def print_torch_info() -> None:
    import torch

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(), name)


def run_training(
    repo: Path,
    data: Path,
    out: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    models: str,
) -> None:
    marker = repo / "problem2" / "colab_train_all.py"
    if not marker.is_file():
        raise SystemExit(f"Missing {marker}. Set --repo to the folder that contains problem2/.")

    if not data.is_file():
        raise SystemExit(f"Missing {data}. Upload TrainingNames.txt or run name generation.")

    cmd = [
        sys.executable,
        "-m",
        "problem2.colab_train_all",
        "--data",
        str(data),
        "--out",
        str(out),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--embed-dim",
        str(embed_dim),
        "--hidden-dim",
        str(hidden_dim),
        "--num-layers",
        str(num_layers),
        "--models",
        models,
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(repo))
    print("\nDone. Output folder:", out)


def upload_training_names_via_colab(destination: str | Path = "/content/TrainingNames.txt") -> Path:
    """Colab only: upload TrainingNames.txt from your machine."""
    from google.colab import files

    uploaded = files.upload()
    if "TrainingNames.txt" not in uploaded:
        raise SystemExit('Expected a file named "TrainingNames.txt" in the upload.')
    dest = Path(destination)
    dest.write_bytes(uploaded["TrainingNames.txt"])
    print("OK:", dest.resolve())
    return dest


def generate_training_names_gemini_colab(
    repo: Path,
    destination: str | Path = "/content/TrainingNames.txt",
    n: int = 1000,
    batch_size: int = 250,
    timeout_sec: int = 240,
    heartbeat_sec: int = 20,
) -> None:
    """Colab only: set GOOGLE_API_KEY via Colab Secrets (userdata) before calling."""
    import os

    from google.colab import userdata

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "google-genai", "python-dotenv"])
    os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "problem2.generate_training_names",
            "-n",
            str(n),
            "--batch-size",
            str(batch_size),
            "--out",
            str(destination),
            "--timeout-sec",
            str(timeout_sec),
            "--heartbeat-sec",
            str(heartbeat_sec),
        ],
        cwd=str(repo),
    )


def main() -> None:
    default_repo = _default_repo()
    ap = argparse.ArgumentParser(description="Train Problem 2 models (Colab GPU driver).")
    ap.add_argument("--repo", type=Path, default=default_repo, help="Repo root (contains problem2/)")
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="TrainingNames.txt (default: TrainingNames.txt or problem2/data/TrainingNames.txt under --repo)",
    )
    ap.add_argument("--out", type=Path, default=None, help="Output dir (default: <repo>/problem2_colab_out)")
    ap.add_argument("--install-deps", action="store_true", help="pip install matplotlib torch")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--models", default="rnn,blstm,attn")
    args = ap.parse_args()

    repo = args.repo.resolve()
    data = _resolve_data_path(repo, args.data)
    out = (args.out or (repo / "problem2_colab_out")).resolve()

    if args.install_deps:
        install_matplotlib_torch()

    print_torch_info()
    run_training(
        repo=repo,
        data=data,
        out=out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        models=args.models,
    )


if __name__ == "__main__":
    main()
