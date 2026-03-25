"""
Legacy entry point: I moved the IITJ corpus code to `problem1/`.

Run:
  uv run python -m problem1.build_corpus
  uv run python -m problem1.build_corpus --no-crawl --rebuild
"""

from __future__ import annotations

from problem1.build_corpus import build_corpus, corpus_stats, main as build_main
from problem1.config import CORPUS_CACHE, DATA_DIR, OUT_DIR, PDF_DIR

# Older scripts expect these names on `task1`.
__all__ = [
    "build_corpus",
    "corpus_stats",
    "CORPUS_CACHE",
    "DATA_DIR",
    "OUT_DIR",
    "PDF_DIR",
]


if __name__ == "__main__":
    build_main()
