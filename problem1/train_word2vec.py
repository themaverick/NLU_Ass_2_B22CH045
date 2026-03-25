"""
Problem 1 Task 2: I train CBOW and Skip-gram on the IITJ corpus.

I sweep at least two values for each category: embedding size (vector_size),
context window (window), and negative samples (negative). I save a CSV and a short report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

from problem1.config import CORPUS_META, OUT_DIR
from problem1.w2v_common import TrainConfig, load_sentences, save_model, train_and_evaluate

RESULTS_CSV = OUT_DIR / "w2v_experiments.csv"
REPORT_MD = OUT_DIR / "problem1_task2_report.md"
EXPERIMENT_JSON = OUT_DIR / "w2v_experiment_manifest.json"


def _best_by(rows: list[dict], arch: str, key: str) -> dict | None:
    subset = [r for r in rows if r["architecture"] == arch]
    if not subset:
        return None
    return max(subset, key=lambda r: r[key])


def run_experiment_grid(
    sentences: list[list[str]],
    vector_sizes: list[int],
    windows: list[int],
    negatives: list[int],
    epochs: int,
    workers: int,
    seed: int,
    save_models: bool,
) -> list[dict]:
    rows: list[dict] = []
    configs: list[TrainConfig] = []
    for vs, win, neg in product(vector_sizes, windows, negatives):
        configs.append(TrainConfig("CBOW", 0, vs, win, neg, epochs))
        configs.append(TrainConfig("Skip-gram", 1, vs, win, neg, epochs))

    manifest: list[dict] = []
    for cfg in configs:
        row, model = train_and_evaluate(sentences, cfg, workers, seed)
        rows.append(row)
        print(
            f"[{cfg.architecture}] d={cfg.vector_size} win={cfg.window} neg={cfg.negative} "
            f"loss={row['training_loss']} google={row['google_analogy_acc']:.4f} "
            f"domain={row['domain_analogy_acc']:.4f}"
        )
        manifest.append({**row, "checkpoint": None})
        if save_models:
            safe = f"{cfg.architecture}_{cfg.vector_size}_{cfg.window}_{cfg.negative}".replace(
                "-", ""
            )
            ckpt = OUT_DIR / "w2v_checkpoints" / f"{safe}.model"
            save_model(model, ckpt)
            manifest[-1]["checkpoint"] = str(ckpt)
        del model

    EXPERIMENT_JSON.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_JSON.write_text(json.dumps({"runs": manifest}, indent=2), encoding="utf-8")
    return rows


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_report(rows: list[dict], epochs: int) -> None:
    n_docs = n_tok = vocab_n = "?"
    if CORPUS_META.is_file():
        meta = json.loads(CORPUS_META.read_text(encoding="utf-8"))
        n_docs = meta.get("num_documents", "?")
        n_tok = meta.get("num_tokens", "?")
        vocab_n = meta.get("vocabulary_size", "?")

    best_cbow_g = _best_by(rows, "CBOW", "google_analogy_acc") if rows else None
    best_sg_g = _best_by(rows, "Skip-gram", "google_analogy_acc") if rows else None
    best_cbow_d = _best_by(rows, "CBOW", "domain_analogy_acc") if rows else None
    best_sg_d = _best_by(rows, "Skip-gram", "domain_analogy_acc") if rows else None

    def fmt_run(r: dict | None, metric: str) -> str:
        if not r:
            return "N/A"
        return (
            f"`d={r['vector_size']}, window={r['window']}, negative={r['negative']}` "
            f"({metric}={r[metric]:.4f})"
        )

    cols = [
        "architecture",
        "vector_size",
        "window",
        "negative",
        "training_loss",
        "google_analogy_acc",
        "domain_analogy_acc",
        "domain_pair_sim",
        "domain_pairs_used",
        "train_seconds",
        "vocab_size",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines_tb = [header, sep]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if c in ("training_loss", "google_analogy_acc", "domain_analogy_acc", "domain_pair_sim"):
                cells.append(f"{v:.6f}" if isinstance(v, (int, float)) else str(v))
            else:
                cells.append(str(v))
        lines_tb.append("| " + " | ".join(cells) + " |")

    report = f"""# Problem 1 — Task 2: Word2Vec experiments

## Data (from `corpus_meta.json`)

| Statistic | Value |
| --- | ---: |
| Documents | {n_docs} |
| Tokens | {n_tok} |
| Vocabulary (types) | {vocab_n} |

## Design

I train **CBOW** (`sg=0`) and **Skip-gram** (`sg=1`) with **negative sampling**. I vary three categories with **at least two settings each**:

- **Embedding size** (`vector_size`): {sorted({r['vector_size'] for r in rows})}
- **Context window** (`window`): {sorted({r['window'] for r in rows})}
- **Negative samples** (`negative`): {sorted({r['negative'] for r in rows})}

That yields a full factorial over those grids (each architecture). I fix `min_count=2`, `epochs={epochs}`, `ns_exponent=0.75`.

## Results (all runs)

{chr(10).join(lines_tb)}

Full table: `{RESULTS_CSV.name}`. Manifest: `{EXPERIMENT_JSON.name}`.

## Best runs (by intrinsic scores)

| Model | Best on Google analogies | Best on domain analogies |
| --- | --- | --- |
| CBOW | {fmt_run(best_cbow_g, "google_analogy_acc")} | {fmt_run(best_cbow_d, "domain_analogy_acc")} |
| Skip-gram | {fmt_run(best_sg_g, "google_analogy_acc")} | {fmt_run(best_sg_d, "domain_analogy_acc")} |

## Notes

- I compare `training_loss` only across runs with the **same** `negative` and `epochs` (gensim scales loss with noise count).
- Google analogy accuracy is often **low** on a narrow domain corpus; I still use it **relatively** across runs.
- I run Task 3–4 scripts separately; they read the same cached corpus under `problem1/output/`.
"""
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_MD}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Problem 1 Task 2 Word2Vec grid")
    p.add_argument("--rebuild-corpus", action="store_true", help="I delete the corpus pickle first.")
    p.add_argument("--quick", action="store_true", help="I use gensim toy sentences and a tiny grid.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-models",
        action="store_true",
        help="I write each gensim model under output/w2v_checkpoints/ (large).",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    mock = args.quick

    # I use at least two values per category (vector_size, window, negative).
    if mock:
        vector_sizes = [64, 128]
        windows = [3, 6]
        negatives = [5, 10]
        epochs = args.epochs if args.epochs is not None else 8
    else:
        vector_sizes = [128, 256]
        windows = [5, 10]
        negatives = [10, 20]
        epochs = args.epochs if args.epochs is not None else 25

    workers = min(4, max(1, __import__("os").cpu_count() or 1))
    sentences = load_sentences(mock=mock, rebuild_corpus=args.rebuild_corpus)
    if not sentences:
        print("No sentences; build the corpus first.", file=sys.stderr)
        sys.exit(1)

    print(
        f"I train {2 * len(vector_sizes) * len(windows) * len(negatives)} runs "
        f"(mock={mock}, sentences={len(sentences)})"
    )
    t0 = time.perf_counter()
    rows = run_experiment_grid(
        sentences,
        vector_sizes,
        windows,
        negatives,
        epochs,
        workers,
        args.seed,
        args.save_models,
    )
    write_csv(rows)
    write_report(rows, epochs)
    print(f"Wrote {RESULTS_CSV} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
