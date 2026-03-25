"""Helpers for Task 3 neighbors and analogies."""

from __future__ import annotations

from problem1.word_lists import QUERY_ALIASES, ANALOGY_TRIPLETS


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


def resolve_analogy_tokens(wv, triplet: tuple[str, str, str]) -> tuple[str, str, str] | None:
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


def analogy_triplets() -> list[tuple[str, str, str]]:
    return list(ANALOGY_TRIPLETS)
