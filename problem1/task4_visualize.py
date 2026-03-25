"""
Problem 1 Task 4: I plot L2-normalized embeddings with PCA and t-SNE (CBOW vs Skip-gram).

I save PNGs under output/plots/ for your report. I pick a large word pool so clusters read clearly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from problem1.config import OUT_DIR, PLOTS_DIR
from problem1.word_lists import VIZ_WORDS
from problem1.w2v_common import TrainConfig, load_sentences, train_and_evaluate

TASK4_MD = OUT_DIR / "task4_visualization.md"


def _dedupe_preserve(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _word_group(w: str) -> int:
    research = {
        "research",
        "development",
        "theory",
        "practice",
        "science",
        "engineering",
        "paper",
        "journal",
        "conference",
        "discovery",
        "experiment",
        "innovation",
    }
    people = {
        "student",
        "faculty",
        "professor",
        "scholar",
        "candidate",
        "supervisor",
        "advisor",
        "mentor",
        "director",
        "dean",
    }
    creds = {
        "phd",
        "bachelor",
        "master",
        "undergraduate",
        "graduate",
        "degree",
        "thesis",
        "dissertation",
    }
    assess = {
        "exam",
        "grade",
        "marks",
        "evaluation",
        "credit",
        "curriculum",
        "syllabus",
        "assessment",
        "assignment",
        "examination",
    }
    org = {
        "university",
        "college",
        "department",
        "program",
        "course",
        "lecture",
        "laboratory",
        "seminar",
        "workshop",
        "academic",
        "committee",
        "office",
        "administration",
    }
    tech = {
        "computer",
        "software",
        "hardware",
        "network",
        "database",
        "algorithm",
        "programming",
        "technology",
        "system",
        "model",
    }
    if w in research:
        return 0
    if w in people:
        return 1
    if w in creds:
        return 2
    if w in assess:
        return 3
    if w in org:
        return 4
    if w in tech:
        return 5
    return 6


def _plot_2d(
    coords: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
    colors: np.ndarray | None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    if colors is not None:
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            cmap="tab10",
            alpha=0.88,
            s=70,
            edgecolors="k",
            linewidths=0.35,
        )
        plt.colorbar(sc, ax=ax, label="group id (see task4_visualization.md)")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.88, s=70, edgecolors="k", linewidths=0.35)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.92)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def project_and_plot(wv, words: list[str], arch_name: str, seed: int) -> tuple[str, str]:
    vecs = np.stack([wv[w] for w in words], axis=0)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    groups = np.array([_word_group(w) for w in words])

    pca = PCA(n_components=2, random_state=seed)
    xy_pca = pca.fit_transform(vecs)
    pca_path = PLOTS_DIR / f"task4_pca_{arch_name.lower().replace('-', '')}.png"
    _plot_2d(xy_pca, words, f"PCA — {arch_name}", pca_path, groups)
    var = float(np.sum(pca.explained_variance_ratio_) * 100.0)
    pca_cap = (
        f"**PCA ({arch_name})** — `{pca_path.name}`. "
        f"First two components explain ~**{var:.1f}%** variance (L2-normalized rows). "
        f"Colors mark coarse groups (research, people, credentials, assessment, org, tech, other)."
    )

    perplexity = min(40, max(5, len(words) // 5))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    xy_tsne = tsne.fit_transform(vecs)
    tsne_path = PLOTS_DIR / f"task4_tsne_{arch_name.lower().replace('-', '')}.png"
    _plot_2d(xy_tsne, words, f"t-SNE — {arch_name}", tsne_path, groups)
    tsne_cap = (
        f"**t-SNE ({arch_name})** — `{tsne_path.name}`. "
        f"I set `perplexity={perplexity}`. Distances across far clusters are only qualitative."
    )
    return pca_cap, tsne_cap


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Problem 1 Task 4 PCA/t-SNE plots")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--rebuild-corpus", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--vector-size", type=int, default=200)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--negative", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-words", type=int, default=45, help="I warn if fewer types plot.")
    return p.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    mock = args.quick
    epochs = args.epochs if args.epochs is not None else (8 if mock else 25)
    workers = min(4, max(1, __import__("os").cpu_count() or 1))

    sentences = load_sentences(mock=mock, rebuild_corpus=args.rebuild_corpus)
    if not sentences:
        print("No corpus; run python -m problem1.build_corpus", file=sys.stderr)
        sys.exit(1)

    base = TrainConfig(
        "CBOW",
        0,
        args.vector_size,
        args.window,
        args.negative,
        epochs,
    )
    _, cbow = train_and_evaluate(
        sentences,
        TrainConfig("CBOW", 0, base.vector_size, base.window, base.negative, base.epochs),
        workers,
        args.seed,
    )
    _, sg = train_and_evaluate(
        sentences,
        TrainConfig("Skip-gram", 1, base.vector_size, base.window, base.negative, base.epochs),
        workers,
        args.seed,
    )

    pool = _dedupe_preserve(list(VIZ_WORDS))
    viz_words = sorted({w for w in pool if w in cbow.wv and w in sg.wv})
    if len(viz_words) < args.min_words:
        extra = [w for w, _ in cbow.wv.key_to_index.items()]
        for w in extra:
            if w not in viz_words and w in sg.wv:
                viz_words.append(w)
            if len(viz_words) >= max(args.min_words, 60):
                break
        viz_words = sorted(viz_words)

    pca_cbow, tsne_cbow = project_and_plot(cbow.wv, viz_words, "CBOW", args.seed)
    pca_sg, tsne_sg = project_and_plot(sg.wv, viz_words, "Skip-gram", args.seed)

    md = f"""# Problem 1 — Task 4: Embedding maps

I plotted **{len(viz_words)}** types that appear in **both** vocabularies after pooling `word_lists.VIZ_WORDS` and frequency fallbacks.

## CBOW

{pca_cbow}

{tsne_cbow}

## Skip-gram

{pca_sg}

{tsne_sg}

## How I colored points

| id | Theme |
| ---: | --- |
| 0 | research / discovery |
| 1 | people roles |
| 2 | credentials |
| 3 | assessment |
| 4 | organization / teaching |
| 5 | technology |
| 6 | other |

**Reading tip:** PCA shows global linear structure; t-SNE emphasizes local neighborhoods.
"""
    TASK4_MD.write_text(md, encoding="utf-8")
    print(f"Wrote plots under {PLOTS_DIR}")
    print(f"Wrote {TASK4_MD}")


if __name__ == "__main__":
    main(sys.argv[1:])
