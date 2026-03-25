#!/usr/bin/env python3
"""Build figures for output/ASSIGNMENT_REPORT.md into output/report_assets/."""

from __future__ import annotations

import csv
import pickle
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

_REPO = Path(__file__).resolve().parents[1]
_OUT = Path(__file__).resolve().parent
_ASSETS = _OUT / "report_assets"
_P1_PLOTS = _REPO / "problem1" / "output" / "plots"
_P2_OUT = _REPO / "problem2" / "problem2_colab_out"
_CORPUS = _REPO / "problem1" / "output" / "corpus_tokens.pkl"
_W2V_CSV = _REPO / "problem1" / "output" / "w2v_experiments.csv"


def _corpus_wordcloud() -> None:
    if not _CORPUS.is_file():
        print(f"Skip word cloud (missing {_CORPUS})")
        return
    with _CORPUS.open("rb") as f:
        docs: list[list[str]] = pickle.load(f)
    counts: Counter[str] = Counter()
    for sent in docs:
        counts.update(sent)
    # cap for readability
    top = dict(counts.most_common(400))
    wc = WordCloud(
        width=1400,
        height=700,
        background_color="white",
        max_words=300,
        colormap="viridis",
        prefer_horizontal=0.85,
    ).generate_from_frequencies(top)
    wc.to_file(str(_ASSETS / "corpus_wordcloud.png"))
    print("Wrote corpus_wordcloud.png")


def _w2v_domain_pair_plot() -> None:
    if not _W2V_CSV.is_file():
        print(f"Skip w2v plot (missing {_W2V_CSV})")
        return
    rows: list[dict] = []
    with _W2V_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    def key(r: dict) -> tuple:
        return (int(r["vector_size"]), int(r["window"]), int(r["negative"]))

    configs = sorted({key(r) for r in rows}, key=lambda x: x)
    labels = [f"{d}_w{w}_n{n}" for d, w, n in configs]
    cbow: list[float] = []
    sg: list[float] = []
    for cfg in configs:
        cbow.append(
            float(
                next(
                    r["domain_pair_sim"]
                    for r in rows
                    if r["architecture"] == "CBOW" and key(r) == cfg
                )
            )
        )
        sg.append(
            float(
                next(
                    r["domain_pair_sim"]
                    for r in rows
                    if r["architecture"] == "Skip-gram" and key(r) == cfg
                )
            )
        )

    x = np.arange(len(configs))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, cbow, width=w, label="CBOW", color="#4c72b0")
    ax.bar(x + w / 2, sg, width=w, label="Skip-gram", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean domain pair cosine similarity")
    ax.set_xlabel("vector_size, window, negative")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ASSETS / "w2v_domain_pair_sim.png", dpi=150)
    plt.close(fig)
    print("Wrote w2v_domain_pair_sim.png")


def _copy_plot(src: Path, name: str) -> None:
    if not src.is_file():
        print(f"Skip copy (missing {src})")
        return
    shutil.copy2(src, _ASSETS / name)
    print(f"Copied {name}")


def main() -> None:
    _ASSETS.mkdir(parents=True, exist_ok=True)
    _corpus_wordcloud()
    _w2v_domain_pair_plot()

    for fn in (
        "task4_pca_cbow.png",
        "task4_tsne_cbow.png",
        "task4_pca_skipgram.png",
        "task4_tsne_skipgram.png",
    ):
        _copy_plot(_P1_PLOTS / fn, fn)

    for fn in ("loss_curve_rnn.png", "loss_curve_blstm.png", "loss_curve_attn.png"):
        _copy_plot(_P2_OUT / fn, fn)


if __name__ == "__main__":
    main()
