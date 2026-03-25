"""Paths and crawl settings for Problem 1."""

from __future__ import annotations

from pathlib import Path

# Package root (problem1/)
PROBLEM1_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROBLEM1_ROOT.parent

DATA_DIR = PROBLEM1_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
OUT_DIR = PROBLEM1_ROOT / "output"
PLOTS_DIR = OUT_DIR / "plots"

CORPUS_CACHE = OUT_DIR / "corpus_tokens.pkl"
CORPUS_META = OUT_DIR / "corpus_meta.json"
CRAWL_STATE = OUT_DIR / "crawl_log.json"

DOMAIN_ANALOGIES = PROBLEM1_ROOT / "domain_analogies.txt"

# I fetch these first so academic PDFs land even if the crawl stops early.
PRIORITY_SEED_URLS: list[str] = [
    "https://iitj.ac.in/PageImages/Gallery/07-2025/Regulation_PG_2022-onwards_20022023.pdf",
    "https://iitj.ac.in/PageImages/Gallery/07-2025/CSE-Courses-Details.pdf",
    "https://www.iitj.ac.in/",
    "https://www.iitj.ac.in/m/Index/main-institute?lg=en",
    "https://www.iitj.ac.in/office-of-director/en/office-of-director",
    "https://www.iitj.ac.in/Bachelor-of-Technology/en/Bachelor-of-Technology",
    "https://www.iitj.ac.in/aiot-fab-facility/en/aiot-fab-facility",
    "https://www.iitj.ac.in/health-center/en/health-center",
    "https://www.iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
    "https://www.iitj.ac.in/computer-science-engineering",
    "https://www.iitj.ac.in/computer-science-engineering/en/faculty-achievements",
]

# I cap the crawl so a single run finishes in reasonable time; raise these if you want more text.
MAX_HTML_PAGES = 280
MAX_PDF_DOWNLOADS = 90
REQUEST_TIMEOUT_S = 45
CRAWL_DELAY_S = 0.45

USER_AGENT = "NLU-PA2-Problem1/1.1 (+educational crawl; contact local course staff)"

# I strip these noisy fragments after HTML/PDF extraction.
BOILERPLATE_SUBSTRINGS = [
    "skip to main content",
    "cookie",
    "privacy policy",
    "home",
    "about iitj",
    "contact",
    "copyright",
    "all rights reserved",
]
