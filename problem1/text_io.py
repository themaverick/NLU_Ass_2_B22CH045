"""Fetch pages, read PDF bytes, clean text, and tokenize (same rules as the original Task 1 script)."""

from __future__ import annotations

import io
import re
import string
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup
from langdetect import detect_langs
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from problem1.config import BOILERPLATE_SUBSTRINGS


def mostly_english(text: str, min_prob: float = 0.9) -> bool:
    text = text.strip()
    if len(text) < 50:
        return True
    try:
        scores = detect_langs(text)
        return scores[0].lang == "en" and scores[0].prob >= min_prob
    except Exception:
        return False


def strip_boilerplate(text: str) -> str:
    t = text.lower()
    for phrase in BOILERPLATE_SUBSTRINGS:
        t = t.replace(phrase, " ")
    return t


def clean_raw_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = strip_boilerplate(text)
    return text


def is_pdf_response(url: str, r: requests.Response) -> bool:
    ct = (r.headers.get("Content-Type") or "").lower()
    if "application/pdf" in ct:
        return True
    path = url.split("?", 1)[0].lower()
    return path.endswith(".pdf")


def _html_text_from_response(r: requests.Response) -> str:
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return clean_raw_text(soup.get_text(separator="\n"))


def pdf_pages_from_bytes(data: bytes) -> list[str]:
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            t = clean_raw_text(t)
            if t:
                pages.append(t)
    return pages


def fetch_url_documents(
    url: str,
    timeout: float,
    user_agent: str,
) -> list[tuple[str, str]]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": user_agent})
    r.raise_for_status()
    if is_pdf_response(url, r):
        return [
            (f"{url}#p{i}", page_text)
            for i, page_text in enumerate(pdf_pages_from_bytes(r.content), start=1)
        ]
    text = _html_text_from_response(r)
    return [(url, text)] if text else []


def read_pdf_file(path: Path) -> list[str]:
    return pdf_pages_from_bytes(path.read_bytes())


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned: list[str] = []
    for tok in tokens:
        tok = tok.strip(string.punctuation)
        if not tok:
            continue
        if re.fullmatch(r"\d+", tok):
            continue
        if re.fullmatch(r"[a-z][a-z'-]*", tok):
            cleaned.append(tok)
    return cleaned
