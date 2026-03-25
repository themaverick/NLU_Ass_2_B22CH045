"""
Task 2: Word2Vec (CBOW vs Skip-gram) with negative sampling — hyperparameter grid,
metrics, CSV log, and formal markdown report.

Run from this directory:
  uv run python task2.py
  uv run python task2.py --quick              # fast smoke test (mock corpus)
  uv run python task2.py --rebuild-corpus     # refetch Task 1 corpus, refresh cache
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from gensim.models import Word2Vec
from gensim.test.utils import datapath

from problem1.build_corpus import build_corpus, corpus_stats
from problem1.config import DOMAIN_ANALOGIES as _P1_DOMAIN_ANALOGIES, OUT_DIR

_ROOT = Path(__file__).resolve().parent
OUT = OUT_DIR
CORPUS_CACHE = OUT / "corpus_tokens.pkl"
RESULTS_CSV = OUT / "task2_results.csv"
REPORT_MD = OUT / "TASK2_REPORT.md"
# Preserved when regenerating this report after `task3_4.py` has appended sections.
TASK34_START = "<!-- NLU_TASK3_4_START -->\n"
TASK34_END = "\n<!-- NLU_TASK3_4_END -->"
DOMAIN_ANALOGIES = _P1_DOMAIN_ANALOGIES

# Cosine similarity of domain-related pairs (both words must appear in vocab)
DOMAIN_PAIRS = [
    ("research", "development"),
    ("student", "academic"),
    ("course", "credit"),
    ("faculty", "professor"),
    ("science", "engineering"),
    ("thesis", "dissertation"),
    ("undergraduate", "graduate"),
    ("lecture", "laboratory"),
]


@dataclass(frozen=True)
class TrainConfig:
    architecture: str  # "CBOW" or "Skip-gram"
    sg: int
    vector_size: int
    window: int
    negative: int
    epochs: int


def load_sentences(mock: bool, rebuild_corpus: bool) -> list[list[str]]:
    if mock:
        from gensim.test.utils import common_texts

        return [list(s) for s in common_texts]

    OUT.mkdir(parents=True, exist_ok=True)
    if rebuild_corpus and CORPUS_CACHE.is_file():
        CORPUS_CACHE.unlink()

    if CORPUS_CACHE.is_file():
        with CORPUS_CACHE.open("rb") as f:
            return pickle.load(f)

    docs, _ = build_corpus()
    with CORPUS_CACHE.open("wb") as f:
        pickle.dump(docs, f)
    return docs


def mean_domain_pair_similarity(wv) -> tuple[float, int]:
    sims: list[float] = []
    for a, b in DOMAIN_PAIRS:
        if a in wv and b in wv:
            sims.append(float(wv.similarity(a, b)))
    if not sims:
        return 0.0, 0
    return sum(sims) / len(sims), len(sims)


def evaluate_embeddings(wv) -> dict[str, float]:
    google_path = datapath("questions-words.txt")
    google_score, _ = wv.evaluate_word_analogies(
        google_path, restrict_vocab=40000, case_insensitive=True
    )

    domain_score = 0.0
    if DOMAIN_ANALOGIES.is_file():
        domain_score, _ = wv.evaluate_word_analogies(
            str(DOMAIN_ANALOGIES), restrict_vocab=40000, case_insensitive=True
        )

    pair_sim, n_pairs = mean_domain_pair_similarity(wv)
    return {
        "google_analogy_acc": float(google_score),
        "domain_analogy_acc": float(domain_score),
        "domain_pair_sim": pair_sim,
        "domain_pairs_used": n_pairs,
    }


def train_and_evaluate(
    sentences: list[list[str]],
    cfg: TrainConfig,
    workers: int,
    seed: int,
) -> dict:
    t0 = time.perf_counter()
    model = Word2Vec(
        sentences=sentences,
        vector_size=cfg.vector_size,
        window=cfg.window,
        min_count=2,
        sg=cfg.sg,
        negative=cfg.negative,
        ns_exponent=0.75,
        epochs=cfg.epochs,
        workers=workers,
        seed=seed,
        compute_loss=True,
        sorted_vocab=True,
    )
    train_sec = time.perf_counter() - t0
    loss = float(model.get_latest_training_loss())
    metrics = evaluate_embeddings(model.wv)
    row = {
        "architecture": cfg.architecture,
        "sg": cfg.sg,
        "vector_size": cfg.vector_size,
        "window": cfg.window,
        "negative": cfg.negative,
        "epochs": cfg.epochs,
        "train_seconds": round(train_sec, 2),
        "training_loss": round(loss, 4),
        "vocab_size": len(model.wv),
        "google_analogy_acc": round(metrics["google_analogy_acc"], 6),
        "domain_analogy_acc": round(metrics["domain_analogy_acc"], 6),
        "domain_pair_sim": round(metrics["domain_pair_sim"], 6),
        "domain_pairs_used": metrics["domain_pairs_used"],
    }
    return row, model


def run_grid(
    sentences: list[list[str]],
    vector_sizes: list[int],
    windows: list[int],
    negatives: list[int],
    epochs: int,
    workers: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    configs: list[TrainConfig] = []
    for vs, win, neg in product(vector_sizes, windows, negatives):
        configs.append(TrainConfig("CBOW", 0, vs, win, neg, epochs))
        configs.append(TrainConfig("Skip-gram", 1, vs, win, neg, epochs))

    for cfg in configs:
        row, _model = train_and_evaluate(sentences, cfg, workers, seed)
        rows.append(row)
        print(
            f"[{cfg.architecture}] d={cfg.vector_size} win={cfg.window} neg={cfg.negative} "
            f"-> loss={row['training_loss']} google_acc={row['google_analogy_acc']:.4f} "
            f"domain_acc={row['domain_analogy_acc']:.4f} pair_sim={row['domain_pair_sim']:.4f}"
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _best_by(rows: list[dict], arch: str, key: str) -> dict | None:
    subset = [r for r in rows if r["architecture"] == arch]
    if not subset:
        return None
    return max(subset, key=lambda r: r[key])


def write_report(rows: list[dict], sentences: list[list[str]], epochs: int) -> None:
    stats = corpus_stats(sentences, remove_stopwords_for_vocab=False)
    total_tokens = stats["num_tokens"]
    n_docs = stats["num_documents"]

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

    # Markdown table
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

    report = f"""# Task 2: Word2Vec model training — report

## 1. Objective

Train two **Word2Vec** architectures with **negative sampling** on the Task 1 corpus:

- **CBOW** (`sg=0`) predicts a target word from averaged context embeddings.
- **Skip-gram** (`sg=1`) predicts context words from the target embedding.

For each architecture, systematically vary **embedding dimension** (`vector_size`), **context window** (`window`), and **number of negative samples** (`negative`), record training loss and intrinsic evaluation scores, and compare configurations.

## 2. Data

| Statistic | Value |
| --- | ---: |
| Documents (tokenized units from Task 1) | {n_docs} |
| Total tokens | {total_tokens} |
| Vocabulary (types, Task 1 tokenizer) | {stats['vocabulary_size']} |

Training uses `min_count=2`, `epochs={epochs}`, `sorted_vocab=True`, and the same lowercased token strings as Task 1.

## 3. Methodology

- **Implementation:** `gensim.models.Word2Vec` (Mikolov-style skip-gram / CBOW with negative sampling). Reported `training_loss` is gensim’s running negative-sampling loss; it scales with `negative`, so compare loss **only** across runs with the **same** `negative` (and same `epochs`).
- **Negative sampling:** `negative` controls how many “noise” draws per positive pair; `ns_exponent=0.75` matches the smoothed unigram distribution used in the original work.
- **Embedding dimension:** larger `vector_size` increases representational capacity but, on a **small** corpus, can overfit or fragment the frequency budget across dimensions.
- **Context window:** larger `window` mixes more distant co-occurrences (broader “semantic” context) but dilutes the immediate predictive signal; Skip-gram often benefits more from wide windows than CBOW because each (target, context) pair is a separate training example.
- **Evaluation (intrinsic):**
  - **Google analogy benchmark** (`questions-words.txt` shipped with gensim): many items are **out-of-vocabulary** on a narrow domain corpus, so **absolute accuracy is often near zero**; it still offers a **comparable relative** signal across runs with identical evaluation settings (`restrict_vocab=40000`, case-insensitive).
  - **Domain analogies** (`domain_analogies.txt`): morphology and academic-style relations using vocabulary more likely to appear in institute text.
  - **Domain pair similarity:** mean cosine similarity between hand-picked related pairs (only pairs where both words exist in the model vocabulary are averaged; `domain_pairs_used` records how many contributed).

Full numeric results are in `{RESULTS_CSV.name}`.

## 4. Results (all runs)

{chr(10).join(lines_tb)}

## 5. Analysis

### 5.1 Best configurations (by metric)

| Model | Best on Google analogies | Best on domain analogies |
| --- | --- | --- |
| CBOW | {fmt_run(best_cbow_g, "google_analogy_acc")} | {fmt_run(best_cbow_d, "domain_analogy_acc")} |
| Skip-gram | {fmt_run(best_sg_g, "google_analogy_acc")} | {fmt_run(best_sg_d, "domain_analogy_acc")} |

### 5.2 Embedding dimension (`vector_size`)

Increasing dimensionality raises model capacity. On **small** corpora, very large embeddings can memorize idiosyncratic co-occurrences without improving general analogy or similarity structure; mid-range dimensions (e.g. 100–300) are a common compromise. In your table, compare rows that share the same `window` and `negative` to isolate the effect of `vector_size` on `training_loss` and analogy scores.

### 5.3 Context window (`window`)

A **narrow** window emphasizes syntactic and immediate collocations (e.g. multi-word phrases). A **wide** window pulls in more document-level co-occurrence signal. Skip-gram typically scales better with larger windows because it emits more independent (center, context) training pairs per sentence. CBOW averages context vectors, so an overly large window can **blur** the context representation, sometimes hurting fine-grained prediction.

### 5.4 Negative samples (`negative`)

More negative samples approximate the softmax denominator more sharply and can stabilize training, but each additional negative increases work per positive example. Values around **5–25** are standard; if `training_loss` and analogy metrics plateau, increasing `negative` further may yield diminishing returns.

### 5.5 CBOW vs Skip-gram

**Skip-gram** tends to perform better on **rare words** because it generates more training updates per rare token. **CBOW** is often faster and can be stronger for **frequent** words when data are abundant. On small domain corpora, Skip-gram is frequently the stronger default for semantic retrieval and analogy-style tests, but the winning configuration should be read from your table rather than assumed.

## 6. Conclusion

This experiment grid isolates the effects of **vector size**, **context window**, and **negative sampling** under a fixed tokenizer and corpus. Use the **relative** ordering of `domain_analogy_acc`, `domain_pair_sim`, and `training_loss` (together with qualitative checks such as `model.wv.most_similar`) to choose a deployment configuration for downstream NLU tasks. For publication-style claims, complement intrinsic scores with an **extrinsic** task (e.g. classification or clustering) on the same domain.

---
*Generated by `task2.py`. Re-run after changing `SOURCE_URLS`, using `--rebuild-corpus`, or editing the hyperparameter grids in `task2.py`. If `task3_4.py` has been run, its appendix between HTML comment markers is preserved across Task 2 regenerations.*
"""

    appendix = ""
    if REPORT_MD.is_file():
        old = REPORT_MD.read_text(encoding="utf-8")
        if TASK34_START in old and TASK34_END in old:
            i = old.index(TASK34_START)
            j = old.index(TASK34_END) + len(TASK34_END)
            appendix = old[i:j]

    if appendix:
        report = report.rstrip() + "\n\n" + appendix + "\n"

    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_MD}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 2 Word2Vec grid search + report")
    p.add_argument(
        "--rebuild-corpus",
        action="store_true",
        help="Ignore/delete cached corpus pickle and rebuild from Task 1",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Mock corpus + tiny grid + few epochs (no downloads)",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    mock = args.quick

    if mock:
        vector_sizes = [64, 128]
        windows = [3, 6]
        negatives = [5, 15]
        epochs = args.epochs if args.epochs is not None else 8
    else:
        vector_sizes = [100, 200, 300]
        windows = [3, 6, 10]
        negatives = [5, 15, 25]
        epochs = args.epochs if args.epochs is not None else 25

    workers = min(4, max(1, __import__("os").cpu_count() or 1))

    sentences = load_sentences(mock=mock, rebuild_corpus=args.rebuild_corpus)
    if not sentences:
        print("No sentences; check Task 1 corpus.", file=sys.stderr)
        sys.exit(1)

    print(f"Sentences: {len(sentences)} (mock={mock}, rebuild_corpus={args.rebuild_corpus})")
    rows = run_grid(sentences, vector_sizes, windows, negatives, epochs, workers, args.seed)
    write_csv(rows)
    print(f"Wrote {RESULTS_CSV}")
    write_report(rows, sentences, epochs)


if __name__ == "__main__":
    main(sys.argv[1:])
