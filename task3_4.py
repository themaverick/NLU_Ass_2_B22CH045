"""
Task 3–4: cosine nearest neighbors, word analogies, and 2D embedding plots (PCA + t-SNE).

Trains one CBOW and one Skip-gram model (same hyperparameters) on the Task 1 corpus,
writes CSV/figure artifacts under output/, and merges a markdown appendix into TASK2_REPORT.md.

Run from project root:
  uv sync && uv run python task3_4.py
  uv run python task3_4.py --quick    # mock corpus, few epochs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from problem1.config import OUT_DIR
from task2 import TrainConfig, load_sentences, train_and_evaluate

_ROOT = Path(__file__).resolve().parent
OUT = OUT_DIR
REPORT_MD = OUT / "TASK2_REPORT.md"
NEIGHBORS_CSV = OUT / "task3_neighbors.csv"
APPENDIX_SNIPPET = OUT / "TASK3_TASK4_APPENDIX.md"

MARKER_START = "<!-- NLU_TASK3_4_START -->\n"
MARKER_END = "\n<!-- NLU_TASK3_4_END -->"

# Words for § nearest neighbors (assignment list; duplicate "exam" in spec → single query)
QUERY_WORDS = ["research", "student", "phd", "exam"]

# If the prompt token is OOV, try these alternates (corpus tokenization variants).
QUERY_ALIASES: dict[str, list[str]] = {
    "exam": ["examination", "exams", "mid-semester", "end-semester"],
}

# (a, b, c) encodes analogy A : B :: C : ?  via vector(B) - vector(A) + vector(C)
ANALOGY_TRIPLETS: list[tuple[str, str, str]] = [
    ("ug", "btech", "pg"),
    ("undergraduate", "bachelor", "graduate"),
    ("faculty", "professor", "student"),
    ("course", "credit", "degree"),
    ("science", "engineering", "theory"),
]

# Pool for visualization (filtered to vocab at runtime)
VIZ_WORDS = [
    "research",
    "student",
    "phd",
    "exam",
    "thesis",
    "dissertation",
    "course",
    "credit",
    "faculty",
    "professor",
    "science",
    "engineering",
    "undergraduate",
    "graduate",
    "bachelor",
    "master",
    "lecture",
    "laboratory",
    "theory",
    "practice",
    "development",
    "academic",
    "university",
    "college",
    "degree",
    "program",
    "seminar",
    "workshop",
    "department",
    "paper",
    "journal",
    "conference",
    "scholar",
    "candidate",
    "admission",
    "curriculum",
    "syllabus",
    "grade",
    "marks",
    "evaluation",
    "project",
    "internship",
]


def _alts(w: str) -> list[str]:
    w = w.lower()
    synonyms: dict[str, list[str]] = {
        "ug": ["ug", "undergraduate"],
        "pg": ["pg", "postgraduate", "graduate"],
        "btech": ["btech", "b.tech", "bachelor"],
        "undergraduate": ["undergraduate", "ug"],
        "graduate": ["graduate", "pg", "postgraduate"],
        "bachelor": ["bachelor", "btech", "b.tech"],
    }
    seen: list[str] = []
    for x in [w] + synonyms.get(w, []):
        if x not in seen:
            seen.append(x)
    return seen


def _resolve_analogy_tokens(
    wv, triplet: tuple[str, str, str]
) -> tuple[str, str, str] | None:
    a, b, c = triplet
    for ca in _alts(a):
        if ca not in wv:
            continue
        for cb in _alts(b):
            if cb not in wv:
                continue
            for cc in _alts(c):
                if cc not in wv:
                    continue
                return ca, cb, cc
    return None


def resolve_query_token(wv, word: str) -> str | None:
    if word in wv:
        return word
    for alt in QUERY_ALIASES.get(word, []):
        if alt in wv:
            return alt
    return None


def top_neighbors_table(wv, word: str, topn: int = 5) -> list[tuple[str, float]]:
    tok = resolve_query_token(wv, word)
    if tok is None:
        return []
    return [(t, float(s)) for t, s in wv.most_similar(tok, topn=topn)]


def analogy_top5(wv, a: str, b: str, c: str) -> list[tuple[str, float]]:
    return [
        (t, float(s))
        for t, s in wv.most_similar(positive=[b, c], negative=[a], topn=5)
    ]


def write_neighbors_csv(
    path: Path,
    cbow_wv,
    sg_wv,
    words: list[str],
    topn: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "query_word",
                "resolved_cbow",
                "resolved_skipgram",
                "rank",
                "neighbor_cbow",
                "cosine_cbow",
                "neighbor_skipgram",
                "cosine_skipgram",
            ]
        )
        for qw in words:
            rc = resolve_query_token(cbow_wv, qw)
            rs = resolve_query_token(sg_wv, qw)
            nc = top_neighbors_table(cbow_wv, qw, topn)
            ns = top_neighbors_table(sg_wv, qw, topn)
            for i in range(topn):
                cb = nc[i] if i < len(nc) else ("", "")
                sb = ns[i] if i < len(ns) else ("", "")
                w.writerow(
                    [
                        qw,
                        rc or "",
                        rs or "",
                        i + 1,
                        cb[0],
                        f"{cb[1]:.6f}" if cb[0] else "",
                        sb[0],
                        f"{sb[1]:.6f}" if sb[0] else "",
                    ]
                )


def _plot_2d(
    coords: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
    colors: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    if colors is not None:
        sc = ax.scatter(
            coords[:, 0], coords[:, 1], c=colors, cmap="tab10", alpha=0.85, s=55, edgecolors="k", linewidths=0.3
        )
        plt.colorbar(sc, ax=ax, label="group id")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.85, s=55, edgecolors="k", linewidths=0.3)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def project_and_plot(
    wv,
    words: list[str],
    arch_name: str,
    seed: int,
) -> tuple[str, str]:
    """Returns (pca_caption, tsne_caption) for the report."""
    vecs = np.stack([wv[w] for w in words], axis=0)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    pca = PCA(n_components=2, random_state=seed)
    xy_pca = pca.fit_transform(vecs)
    pca_path = OUT / f"task4_pca_{arch_name.lower().replace('-', '')}.png"
    # Color by coarse groups for readability
    groups = np.array([_word_group(w) for w in words])
    _plot_2d(xy_pca, words, f"PCA — {arch_name}", pca_path, colors=groups)
    var = float(np.sum(pca.explained_variance_ratio_) * 100.0)
    pca_cap = (
        f"**Figure (PCA, {arch_name}).** File `{pca_path.name}`. "
        f"Two principal components capture about **{var:.1f}%** of variance in the selected "
        f"embedding matrix (L2-normalized rows). Points are colored by coarse category "
        f"(research/teaching, people, credentials, assessment, org). "
        f"PCA is linear and preserves global structure; nearby points share major variance directions."
    )

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(words) // 4)),
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    xy_tsne = tsne.fit_transform(vecs)
    tsne_path = OUT / f"task4_tsne_{arch_name.lower().replace('-', '')}.png"
    _plot_2d(xy_tsne, words, f"t-SNE — {arch_name}", tsne_path, colors=groups)
    tsne_cap = (
        f"**Figure (t-SNE, {arch_name}).** File `{tsne_path.name}`. "
        f"Nonlinear projection (perplexity≈{min(30, max(5, len(words) // 4))}) of the same vectors. "
        f"Local neighborhoods are emphasized; distances across disjoint clusters are not strictly comparable."
    )
    return pca_cap, tsne_cap


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
    }
    people = {
        "student",
        "faculty",
        "professor",
        "scholar",
        "candidate",
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
    assess = {"exam", "grade", "marks", "evaluation", "credit", "curriculum", "syllabus"}
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
    return 5


def build_appendix_markdown(
    cfg: TrainConfig,
    cbow: Word2Vec,
    sg: Word2Vec,
    pca_cbow_cap: str,
    pca_sg_cap: str,
    tsne_cbow_cap: str,
    tsne_sg_cap: str,
    analogy_rows: list[dict],
) -> str:
    cbow_wv = cbow.wv
    sg_wv = sg.wv

    lines: list[str] = []
    lines.append("## 7. Task 3: Semantic analysis (cosine similarity)\n")
    lines.append("### 7.1 Setup\n")
    lines.append(
        f"For qualitative analysis we trained **one CBOW** and **one Skip-gram** model with the **same** "
        f"hyperparameters: `vector_size={cfg.vector_size}`, `window={cfg.window}`, `negative={cfg.negative}`, "
        f"`epochs={cfg.epochs}`, `min_count=2`, negative sampling with `ns_exponent=0.75`. "
        f"**Nearest neighbors** use **cosine** similarity via `gensim.models.KeyedVectors.most_similar`.\n"
    )

    lines.append("### 7.2 Top-5 nearest neighbors\n")
    lines.append(
        "If a query is out-of-vocabulary, we try aliases from `QUERY_ALIASES` (e.g. **exam** → **examination**).\n"
    )
    lines.append("| Query | Resolved | Model | Rank | Neighbor | Cosine |")
    lines.append("| --- | --- | --- | ---: | --- | ---: |")
    for qw in QUERY_WORDS:
        for arch, wv in (("CBOW", cbow_wv), ("Skip-gram", sg_wv)):
            rt = resolve_query_token(wv, qw)
            if rt is None:
                lines.append(f"| {qw} | — | {arch} | — | *OOV* | — |")
                continue
            res_col = rt if rt == qw else f"{qw} → {rt}"
            for rank, (tok, sim) in enumerate(top_neighbors_table(wv, qw, 5), start=1):
                lines.append(f"| {qw} | {res_col} | {arch} | {rank} | {tok} | {sim:.4f} |")
    lines.append("")
    lines.append(f"Full long-form table: `{NEIGHBORS_CSV.name}`.\n")

    lines.append("### 7.3 Analogy experiments (vector offset)\n")
    lines.append(
        "For each prompt **A : B :: C : ?** we rank tokens by cosine similarity to **B − A + C**. "
        "Abbreviations (**ug**, **pg**, **btech**) map to the first matching synonym present in the vocabulary "
        "(see `_alts` in `task3_4.py`).\n"
    )
    for row in analogy_rows:
        a, b, c = row["a"], row["b"], row["c"]
        lines.append(f"#### `{a} : {b} :: {c} : ?`\n")
        rc = row.get("resolved_cbow")
        rs = row.get("resolved_sg")
        for model_name, key, res in (
            ("CBOW", "preds_cbow", rc),
            ("Skip-gram", "preds_sg", rs),
        ):
            if res:
                lines.append(
                    f"- **{model_name}** — vectors for "
                    f"({res[0]}, {res[1]}, {res[2]}).\n"
                )
            else:
                lines.append(f"- **{model_name}** — *could not resolve A,B,C in vocabulary*.\n")
            preds = row[key]
            lines.append("| Rank | Prediction | Cosine |")
            lines.append("| ---: | --- | ---: |")
            if not preds:
                lines.append("| — | *no prediction* | — |")
            else:
                for i, (tok, sim) in enumerate(preds, start=1):
                    lines.append(f"| {i} | {tok} | {sim:.4f} |")
            lines.append("")

    lines.append("")
    lines.append("### 7.4 Discussion (semantic plausibility)\n")
    lines.append(
        "- **Neighbors:** On a **small, domain-specific** corpus, neighbors often reflect **document co-occurrence** "
        "(committee names, local collocations) as much as abstract synonymy. CBOW **smooths** context, which can "
        "yield **higher cosine** with broad topical associates; Skip-gram **emphasizes** predictive links and "
        "often surfaces rarer but informative contexts.\n"
    )
    lines.append(
        "- **Analogies:** Offset analogies assume linear relations between embeddings. They succeed when "
        "**A:B** and **C:?** correspond to a **consistent** relation learned from data (e.g. parallel degree "
        "naming). Spelling variants (**ug** vs **undergraduate**) and OOV tokens break analogies; when all "
        "tokens are in-vocabulary, inspect whether top predictions are **paraphrases**, **siblings in a taxonomy**, "
        "or **artifacts** of shared boilerplate.\n"
    )

    lines.append("\n## 8. Task 4: Visualization (PCA and t-SNE)\n")
    lines.append("### 8.1 Word set\n")
    lines.append(
        "We projected **L2-normalized** embeddings for a fixed list of domain-relevant types that appear in "
        "**both** vocabularies (see script `task3_4.py`). Colors mark coarse groups (research, people, credentials, "
        "assessment, organization, other).\n"
    )
    lines.append("### 8.2 Figures and captions\n")
    lines.append("![PCA — CBOW](task4_pca_cbow.png)\n")
    lines.append(pca_cbow_cap + "\n\n")
    lines.append("![PCA — Skip-gram](task4_pca_skipgram.png)\n")
    lines.append(pca_sg_cap + "\n\n")
    lines.append("![t-SNE — CBOW](task4_tsne_cbow.png)\n")
    lines.append(tsne_cbow_cap + "\n\n")
    lines.append("![t-SNE — Skip-gram](task4_tsne_skipgram.png)\n")
    lines.append(tsne_sg_cap + "\n\n")
    lines.append("### 8.3 CBOW vs Skip-gram clustering\n")
    lines.append(
        "- **CBOW** vectors are trained to predict the center from **averaged** context; clusters in PCA often "
        "align with **broad topical blobs** (frequent words dominate the average).\n"
    )
    lines.append(
        "- **Skip-gram** updates each context direction separately, which tends to preserve **finer** relational "
        "structure; t-SNE may show **tighter** micro-clusters of synonyms or role-related words, but can also "
        "separate **low-frequency** types if context evidence is sparse.\n"
    )
    lines.append(
        "- **PCA vs t-SNE:** PCA highlights **global** linear separations; t-SNE highlights **local** neighborhoods. "
        "Use PCA for a coarse layout check and t-SNE for neighborhood structure, without over-interpreting "
        "**between-cluster** t-SNE distances.\n"
    )

    return "\n".join(lines)


def merge_into_report(main_path: Path, appendix: str) -> None:
    if not main_path.is_file():
        print(f"Report not found: {main_path}; wrote appendix only to {APPENDIX_SNIPPET}", file=sys.stderr)
        return
    body = main_path.read_text(encoding="utf-8")
    wrapped = MARKER_START + appendix + MARKER_END
    if MARKER_START in body and MARKER_END in body:
        i = body.index(MARKER_START)
        j = body.index(MARKER_END) + len(MARKER_END)
        body = body[:i] + wrapped + body[j:]
    else:
        body = body.rstrip() + "\n\n" + wrapped + "\n"
    main_path.write_text(body, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3–4 semantic analysis + embedding plots")
    p.add_argument("--quick", action="store_true", help="Mock corpus + few epochs")
    p.add_argument("--rebuild-corpus", action="store_true", help="Rebuild Task 1 corpus cache")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--vector-size", type=int, default=100)
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--negative", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-merge-report",
        action="store_true",
        help="Only write TASK3_TASK4_APPENDIX.md; do not modify TASK2_REPORT.md",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    mock = args.quick
    epochs = args.epochs if args.epochs is not None else (8 if mock else 25)
    workers = min(4, max(1, __import__("os").cpu_count() or 1))

    sentences = load_sentences(mock=mock, rebuild_corpus=args.rebuild_corpus)
    if not sentences:
        print("No sentences; run Task 1 / task2 first.", file=sys.stderr)
        sys.exit(1)

    base = TrainConfig(
        architecture="CBOW",
        sg=0,
        vector_size=args.vector_size,
        window=args.window,
        negative=args.negative,
        epochs=epochs,
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

    cbow_wv = cbow.wv
    sg_wv = sg.wv

    write_neighbors_csv(NEIGHBORS_CSV, cbow_wv, sg_wv, QUERY_WORDS, 5)

    analogy_rows: list[dict] = []
    for trip in ANALOGY_TRIPLETS:
        resolved_c = _resolve_analogy_tokens(cbow_wv, trip)
        resolved_s = _resolve_analogy_tokens(sg_wv, trip)
        a0, b0, c0 = trip
        row = {
            "a": a0,
            "b": b0,
            "c": c0,
            "resolved_cbow": resolved_c,
            "resolved_sg": resolved_s,
            "preds_cbow": analogy_top5(cbow_wv, *resolved_c) if resolved_c else [],
            "preds_sg": analogy_top5(sg_wv, *resolved_s) if resolved_s else [],
        }
        analogy_rows.append(row)

    viz_words = sorted({w for w in VIZ_WORDS if w in cbow_wv and w in sg_wv})
    if len(viz_words) < 8:
        # fallback: most frequent content words
        viz_words = [w for w, _ in cbow_wv.key_to_index.items()]
        viz_words = sorted(viz_words[:40])

    pca_cbow_cap, tsne_cbow_cap = project_and_plot(cbow_wv, viz_words, "CBOW", args.seed)
    pca_sg_cap, tsne_sg_cap = project_and_plot(sg_wv, viz_words, "Skip-gram", args.seed)

    appendix = build_appendix_markdown(
        base,
        cbow,
        sg,
        pca_cbow_cap,
        pca_sg_cap,
        tsne_cbow_cap,
        tsne_sg_cap,
        analogy_rows,
    )
    APPENDIX_SNIPPET.write_text(appendix, encoding="utf-8")
    print(f"Wrote {APPENDIX_SNIPPET}")
    print(f"Wrote {NEIGHBORS_CSV}")
    print(f"Wrote plots under {OUT}/task4_*.png")

    if not args.no_merge_report:
        merge_into_report(REPORT_MD, appendix)
        print(f"Merged Task 3–4 appendix into {REPORT_MD}")


if __name__ == "__main__":
    main(sys.argv[1:])
