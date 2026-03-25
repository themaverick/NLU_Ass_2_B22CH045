#!/usr/bin/env python3
"""Run novelty/diversity evaluation for all trained checkpoints in an output folder."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate all checkpoint_*.pt in a directory.")
    ap.add_argument("--train-data", type=Path, required=True, help="TrainingNames.txt used for training")
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("problem2/problem2_colab_out"),
        help="Folder containing checkpoint_rnn.pt, checkpoint_blstm.pt, checkpoint_attn.pt",
    )
    ap.add_argument("--n", type=int, default=500, help="Samples per model")
    ap.add_argument("--temperature", type=float, default=0.85)
    args = ap.parse_args()

    base = args.checkpoint_dir.resolve()
    for kind in ("rnn", "blstm", "attn"):
        ckpt = base / f"checkpoint_{kind}.pt"
        if not ckpt.is_file():
            print(f"Skip (missing): {ckpt}", file=sys.stderr)
            continue
        out_json = base / f"eval_{kind}.json"
        cmd = [
            sys.executable,
            "-m",
            "problem2.evaluate",
            "--checkpoint",
            str(ckpt),
            "--train-data",
            str(args.train_data),
            "--n",
            str(args.n),
            "--temperature",
            str(args.temperature),
            "--out-json",
            str(out_json),
        ]
        print(" ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(_REPO_ROOT))


if __name__ == "__main__":
    main()
