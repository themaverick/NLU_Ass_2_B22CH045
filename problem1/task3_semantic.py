"""
Problem 1 Task 3: I print cosine neighbors and vector-offset analogies for CBOW vs Skip-gram.

I train one model per architecture (same hyperparameters) unless you pass --load paths.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from problem1.config import OUT_DIR
from problem1.semantic_utils import (
    analogy_top5,
    analogy_triplets,
    resolve_analogy_tokens,
    resolve_query_token,
    top_neighbors_table,
)
from problem1.word_lists import QUERY_WORDS
from problem1.w2v_common import TrainConfig, load_sentences, train_and_evaluate

NEIGHBORS_CSV = OUT_DIR / "task3_neighbors.csv"
TASK3_MD = OUT_DIR / "task3_semantic_analysis.md"


def write_neighbors_csv(path: Path, cbow_wv, sg_wv, words: list[str], topn: int) -> None:
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


def build_markdown(
    cfg: TrainConfig,
    cbow_wv,
    sg_wv,
    analogy_rows: list[dict],
) -> str:
    lines: list[str] = []
    lines.append("# Problem 1 — Task 3: Semantic analysis\n\n")
    lines.append(
        f"I trained **CBOW** and **Skip-gram** with `vector_size={cfg.vector_size}`, "
        f"`window={cfg.window}`, `negative={cfg.negative}`, `epochs={cfg.epochs}`.\n\n"
    )
    lines.append("## Top-5 cosine neighbors\n\n")
    lines.append("| Query | Resolved | Model | Rank | Neighbor | Cosine |\n")
    lines.append("| --- | --- | --- | ---: | --- | ---: |\n")
    for qw in QUERY_WORDS:
        for arch, wv in (("CBOW", cbow_wv), ("Skip-gram", sg_wv)):
            rt = resolve_query_token(wv, qw)
            if rt is None:
                lines.append(f"| {qw} | — | {arch} | — | *OOV* | — |\n")
                continue
            res_col = rt if rt == qw else f"{qw} → {rt}"
            for rank, (tok, sim) in enumerate(top_neighbors_table(wv, qw, 5), start=1):
                lines.append(f"| {qw} | {res_col} | {arch} | {rank} | {tok} | {sim:.4f} |\n")
    lines.append("\n## Analogies (vector offset)\n\n")
    for row in analogy_rows:
        a, b, c = row["a"], row["b"], row["c"]
        lines.append(f"### `{a} : {b} :: {c} : ?`\n\n")
        for model_name, key, resolved in (
            ("CBOW", "preds_cbow", row.get("resolved_cbow")),
            ("Skip-gram", "preds_sg", row.get("resolved_sg")),
        ):
            lines.append(f"**{model_name}** ")
            if resolved:
                lines.append(f"(tokens `{resolved[0]}`, `{resolved[1]}`, `{resolved[2]}`)\n\n")
            else:
                lines.append("— *could not resolve A,B,C*\n\n")
            preds = row[key]
            lines.append("| Rank | Token | Cosine |\n| ---: | --- | ---: |\n")
            if not preds:
                lines.append("| — | — | — |\n")
            else:
                for i, (tok, sim) in enumerate(preds, start=1):
                    lines.append(f"| {i} | {tok} | {sim:.4f} |\n")
            lines.append("\n")
    lines.append(
        "**Takeaway:** On a domain corpus, neighbors track **co-occurrence**; analogies need all tokens in vocabulary.\n"
    )
    return "".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Problem 1 Task 3 neighbors + analogies")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--rebuild-corpus", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--vector-size", type=int, default=200)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--negative", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
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

    cbow_wv = cbow.wv
    sg_wv = sg.wv

    write_neighbors_csv(NEIGHBORS_CSV, cbow_wv, sg_wv, QUERY_WORDS, 5)

    analogy_rows: list[dict] = []
    for trip in analogy_triplets():
        rc = resolve_analogy_tokens(cbow_wv, trip)
        rs = resolve_analogy_tokens(sg_wv, trip)
        a0, b0, c0 = trip
        analogy_rows.append(
            {
                "a": a0,
                "b": b0,
                "c": c0,
                "resolved_cbow": rc,
                "resolved_sg": rs,
                "preds_cbow": analogy_top5(cbow_wv, *rc) if rc else [],
                "preds_sg": analogy_top5(sg_wv, *rs) if rs else [],
            }
        )

    md = build_markdown(base, cbow_wv, sg_wv, analogy_rows)
    TASK3_MD.parent.mkdir(parents=True, exist_ok=True)
    TASK3_MD.write_text(md, encoding="utf-8")
    print(f"Wrote {NEIGHBORS_CSV}")
    print(f"Wrote {TASK3_MD}")


if __name__ == "__main__":
    main(sys.argv[1:])
