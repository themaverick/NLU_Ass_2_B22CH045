"""I build the tokenized document list: seeds, crawled pages, and every pdf under data/pdfs."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import nltk

from problem1.config import (
    CORPUS_CACHE,
    CORPUS_META,
    CRAWL_STATE,
    DATA_DIR,
    OUT_DIR,
    PDF_DIR,
    PRIORITY_SEED_URLS,
    REQUEST_TIMEOUT_S,
    USER_AGENT,
)
from problem1.crawl import run_iitj_crawl, write_log
from problem1.text_io import (
    fetch_url_documents,
    mostly_english,
    read_pdf_file,
    tokenize,
)


def _ensure_nltk() -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)


def corpus_stats(tokenized_docs: list[list[str]], remove_stopwords_for_vocab: bool = False) -> dict:
    from nltk.corpus import stopwords

    all_tokens = [t for doc in tokenized_docs for t in doc]
    sw = set(stopwords.words("english"))
    if remove_stopwords_for_vocab:
        types = {t for t in all_tokens if t not in sw}
    else:
        types = set(all_tokens)
    return {
        "num_documents": len(tokenized_docs),
        "num_tokens": len(all_tokens),
        "vocabulary_size": len(types),
    }


def build_corpus(
    *,
    run_crawl: bool = True,
    extra_seeds: list[str] | None = None,
) -> tuple[list[list[str]], list[str]]:
    _ensure_nltk()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = list(PRIORITY_SEED_URLS)
    if extra_seeds:
        seeds.extend(extra_seeds)

    if run_crawl:
        log = run_iitj_crawl(seeds)
        write_log(CRAWL_STATE, log, seeds)
        print(
            f"Crawl: html={log.html_pages_fetched} pdf_saved={log.pdf_files_saved} "
            f"pdf_urls_seen={log.pdf_urls_seen} errors={len(log.errors)}"
        )

    tokenized_docs: list[list[str]] = []
    doc_sources: list[str] = []

    for url in seeds:
        try:
            for label, raw in fetch_url_documents(url, REQUEST_TIMEOUT_S, USER_AGENT):
                if not mostly_english(raw):
                    continue
                toks = tokenize(raw)
                if toks:
                    tokenized_docs.append(toks)
                    doc_sources.append(label)
        except Exception as e:
            print(f"skip seed {url}: {e}", file=sys.stderr)

    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        for i, page_text in enumerate(read_pdf_file(pdf_path), start=1):
            if not mostly_english(page_text):
                continue
            toks = tokenize(page_text)
            if toks:
                tokenized_docs.append(toks)
                doc_sources.append(f"{pdf_path.name}#p{i}")

    return tokenized_docs, doc_sources


def save_corpus(
    docs: list[list[str]],
    sources: list[str],
    cache_path: Path,
    meta_path: Path,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(docs, f)
    stats = corpus_stats(docs)
    meta = {
        "num_documents": stats["num_documents"],
        "num_tokens": stats["num_tokens"],
        "vocabulary_size": stats["vocabulary_size"],
        "sources_sample": sources[:30],
        "total_sources": len(sources),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {cache_path} ({stats['num_documents']} docs, {stats['num_tokens']} tokens)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Problem 1 IITJ corpus (pickle + meta json).")
    p.add_argument(
        "--no-crawl",
        action="store_true",
        help="I only use seed URLs and PDFs already in data/pdfs (no BFS).",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="I overwrite the cached pickle even if it exists.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if CORPUS_CACHE.is_file() and not args.rebuild:
        print(f"I reuse existing corpus: {CORPUS_CACHE} (pass --rebuild to refresh)")
        return

    docs, sources = build_corpus(run_crawl=not args.no_crawl)
    if not docs:
        print("No documents; check network and data/pdfs.", file=sys.stderr)
        sys.exit(1)
    save_corpus(docs, sources, CORPUS_CACHE, CORPUS_META)


if __name__ == "__main__":
    main()
