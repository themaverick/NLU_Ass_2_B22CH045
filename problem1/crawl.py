"""I crawl html under iitj.ac.in, collect pdf links, and save pdfs locally."""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from problem1.config import (
    CRAWL_DELAY_S,
    MAX_HTML_PAGES,
    MAX_PDF_DOWNLOADS,
    PDF_DIR,
    REQUEST_TIMEOUT_S,
    USER_AGENT,
)


def _netloc_ok(netloc: str) -> bool:
    n = netloc.lower()
    return n == "iitj.ac.in" or n.endswith(".iitj.ac.in") or n == "www.iitj.ac.in"


def _normalize_url(url: str) -> str:
    url = url.split("#", 1)[0].strip()
    if url.startswith("//"):
        return "https:" + url
    return url


def _same_site(url: str) -> bool:
    try:
        return _netloc_ok(urlparse(url).netloc)
    except Exception:
        return False


@dataclass
class CrawlLog:
    html_pages_fetched: int
    pdf_urls_seen: int
    pdf_files_saved: int
    errors: list[str]


def extract_links(html: str, base_url: str) -> tuple[list[str], list[str]]:
    soup = BeautifulSoup(html, "lxml")
    html_urls: list[str] = []
    pdf_urls: list[str] = []
    for a in soup.find_all("a", href=True):
        raw = a["href"].strip()
        if raw.startswith(("mailto:", "javascript:", "tel:")):
            continue
        abs_url = urljoin(base_url, raw)
        abs_url = _normalize_url(abs_url)
        low = abs_url.lower()
        if low.endswith(".pdf") or "/pageimages/" in low and ".pdf" in low:
            pdf_urls.append(abs_url)
        elif abs_url.startswith("http") and _same_site(abs_url):
            html_urls.append(abs_url)
    return html_urls, pdf_urls


def download_pdf(url: str, dest_dir: Path, session: requests.Session, timeout: float) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    r = session.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    if "pdf" not in (r.headers.get("Content-Type") or "").lower() and not url.lower().endswith(".pdf"):
        return None
    tail = urlparse(url).path.rsplit("/", 1)[-1] or "doc.pdf"
    tail = re.sub(r"[^\w.\-]", "_", tail)[:180]
    if not tail.lower().endswith(".pdf"):
        tail += ".pdf"
    path = dest_dir / tail
    if path.exists():
        return path
    data = r.content
    if len(data) < 200:
        return None
    path.write_bytes(data)
    return path


def run_iitj_crawl(
    seed_urls: list[str],
    max_html_pages: int = MAX_HTML_PAGES,
    max_pdf_downloads: int = MAX_PDF_DOWNLOADS,
) -> CrawlLog:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    errors: list[str] = []
    html_count = 0
    pdf_seen = 0
    pdf_saved = 0

    seen_html: set[str] = set()
    seen_pdf: set[str] = set()
    q: deque[tuple[str, int]] = deque()

    for u in seed_urls:
        u = _normalize_url(u)
        if not u.startswith("http") or not _same_site(u):
            continue
        if u.lower().endswith(".pdf"):
            if u in seen_pdf or pdf_saved >= max_pdf_downloads:
                continue
            seen_pdf.add(u)
            pdf_seen += 1
            try:
                time.sleep(CRAWL_DELAY_S)
                p = download_pdf(u, PDF_DIR, session, REQUEST_TIMEOUT_S)
                if p is not None:
                    pdf_saved += 1
            except Exception as e:
                errors.append(f"seed pdf {u}: {e!s}")
            continue
        q.append((u, 0))

    while q and html_count < max_html_pages and pdf_saved < max_pdf_downloads:
        url, depth = q.popleft()
        if url in seen_html:
            continue
        if not _same_site(url):
            continue
        low = url.lower()
        if low.endswith(".pdf"):
            if url not in seen_pdf and pdf_saved < max_pdf_downloads:
                seen_pdf.add(url)
                pdf_seen += 1
                try:
                    time.sleep(CRAWL_DELAY_S)
                    p = download_pdf(url, PDF_DIR, session, REQUEST_TIMEOUT_S)
                    if p is not None:
                        pdf_saved += 1
                except Exception as e:
                    errors.append(f"pdf {url}: {e!s}")
            continue

        seen_html.add(url)
        html_count += 1
        try:
            time.sleep(CRAWL_DELAY_S)
            r = session.get(url, timeout=REQUEST_TIMEOUT_S)
            r.raise_for_status()
            ct = (r.headers.get("Content-Type") or "").lower()
            if "pdf" in ct:
                if url not in seen_pdf and pdf_saved < max_pdf_downloads:
                    seen_pdf.add(url)
                    pdf_seen += 1
                    try:
                        p = download_pdf(url, PDF_DIR, session, REQUEST_TIMEOUT_S)
                        if p is not None:
                            pdf_saved += 1
                    except Exception as e:
                        errors.append(f"pdf-inline {url}: {e!s}")
                continue
            if "html" not in ct and "text" not in ct:
                continue
            links, pdfs = extract_links(r.text, url)
            for pu in pdfs:
                if pu in seen_pdf or pdf_saved >= max_pdf_downloads:
                    continue
                seen_pdf.add(pu)
                pdf_seen += 1
                try:
                    time.sleep(CRAWL_DELAY_S)
                    p = download_pdf(pu, PDF_DIR, session, REQUEST_TIMEOUT_S)
                    if p is not None:
                        pdf_saved += 1
                except Exception as e:
                    errors.append(f"pdf {pu}: {e!s}")
            if depth < 4:
                for lu in links:
                    if lu not in seen_html and _same_site(lu):
                        q.append((lu, depth + 1))
        except Exception as e:
            errors.append(f"html {url}: {e!s}")

    return CrawlLog(
        html_pages_fetched=html_count,
        pdf_urls_seen=pdf_seen,
        pdf_files_saved=pdf_saved,
        errors=errors[:80],
    )


def write_log(path: Path, log: CrawlLog, seeds: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seeds": seeds, **asdict(log)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
