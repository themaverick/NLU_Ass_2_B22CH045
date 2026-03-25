"""Shared Word2Vec training helpers and evaluation (Problem 1 Task 2)."""

from __future__ import annotations

import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from gensim.models import Word2Vec
from gensim.test.utils import datapath

from problem1.config import CORPUS_CACHE, DOMAIN_ANALOGIES, OUT_DIR

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
    architecture: str
    sg: int
    vector_size: int
    window: int
    negative: int
    epochs: int


def load_sentences(mock: bool, rebuild_corpus: bool) -> list[list[str]]:
    if mock:
        from gensim.test.utils import common_texts

        return [list(s) for s in common_texts]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if rebuild_corpus and CORPUS_CACHE.is_file():
        CORPUS_CACHE.unlink()

    if CORPUS_CACHE.is_file():
        with CORPUS_CACHE.open("rb") as f:
            return pickle.load(f)

    print("I need a corpus; run: python -m problem1.build_corpus", file=sys.stderr)
    sys.exit(1)


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
) -> tuple[dict, Word2Vec]:
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


def save_model(model: Word2Vec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
