#!/usr/bin/env python3
"""
TASK-0: Build TrainingNames.txt with unique Indian **first names** only.

I call Gemini in batches (default 250 names per request), add each name to a **set** for
deduplication, and discard repeats until I reach the target count.

Uses GOOGLE_API_KEY from the environment (loads .env from project root if present).
Use --mock to synthesize offline (no API).

Do not commit API keys. This script never prints the key.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

_T = TypeVar("_T")


def _run_with_heartbeat(
    fn: Callable[[], _T],
    *,
    verbose: bool,
    heartbeat_sec: int,
    msg: str,
) -> _T:
    """I print `msg` every `heartbeat_sec` while `fn()` blocks (e.g. Gemini HTTP call)."""
    done = threading.Event()

    def loop() -> None:
        while not done.wait(timeout=heartbeat_sec):
            if verbose:
                print(msg, flush=True)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        done.set()

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _ROOT / "problem2" / "data" / "TrainingNames.txt"
_DEFAULT_BATCH = 250


def _load_dotenv() -> None:
    env_path = _ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        pass


def normalize_first_name(raw: str) -> str | None:
    """I keep a single token: Latin letters, hyphen, apostrophe; 2–40 chars."""
    s = raw.strip().strip('"').strip("'")
    if not s:
        return None
    # I take the first word only (first name).
    token = s.split()[0] if s.split() else s
    token = token.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]{1,39}", token):
        return None
    return token


def _parse_name_lines(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^\d+[\).\s]+", "", line)
        parts = re.split(r"[,;|]", line)
        for p in parts:
            n = normalize_first_name(p)
            if n:
                out.append(n)
    return out


def _gemini_api_key() -> str | None:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def fetch_gemini(
    target: int,
    model_name: str,
    batch_size: int,
    *,
    verbose: bool = True,
    timeout_ms: int = 180_000,
    heartbeat_sec: int = 30,
) -> list[str]:
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as e:
        raise RuntimeError(
            "Install the new SDK: pip install google-genai (or uv sync --extra problem2)"
        ) from e

    key = _gemini_api_key()
    if not key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY. Use --mock or set the key in your environment."
        )

    client = genai.Client(
        api_key=key,
        http_options=genai_types.HttpOptions(timeout=timeout_ms),
    )
    seen: set[str] = set()
    ordered: list[str] = []
    batch_idx = 0

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log(
        f"Target: {target} unique first names (batch_size={batch_size}, model={model_name}, "
        f"HTTP timeout={timeout_ms // 1000}s)"
    )

    while len(ordered) < target:
        batch_idx += 1
        remaining = target - len(ordered)
        # I always ask for a full batch_size when the overall target is large enough, so the model
        # keeps seeing a big list request (easier to get diverse names at the end). I only shrink
        # when target < batch_size (small run) or remaining would be 0.
        if target < batch_size:
            need = remaining
        else:
            need = batch_size
        log(
            f"[batch {batch_idx}] {len(ordered)}/{target} unique so far — "
            f"requesting {need} from Gemini (waiting on network; heartbeat every {heartbeat_sec}s)…"
        )
        prompt = f"""Generate exactly {need} distinct realistic **Indian first names only** (given names).

Rules:
- **Single word per line** — first name only. No surnames, no family names, no titles.
- Latin script. Mix regions and religions plausibly.
- Plain text: one name per line, no numbers, bullets, or commas.
- Do not repeat any name in your reply.
"""

        def do_generate():
            return client.models.generate_content(model=model_name, contents=prompt)

        try:
            resp = _run_with_heartbeat(
                do_generate,
                verbose=verbose,
                heartbeat_sec=heartbeat_sec,
                msg=(
                    f"[batch {batch_idx}] still waiting on Gemini (no response yet; "
                    f"timeout in at most {timeout_ms // 1000}s)…"
                ),
            )
        except Exception as e:
            raise RuntimeError(
                f"Gemini request failed: {e!s}. "
                "Check network/VPN, API key, and model id. "
                "Try increasing --timeout-sec or use --mock."
            ) from e
        text = (getattr(resp, "text", None) or "").strip()
        batch = _parse_name_lines(text)
        before = len(ordered)
        for n in batch:
            k = n.lower()
            if k not in seen:
                seen.add(k)
                ordered.append(n)
                if len(ordered) >= target:
                    break
        added = len(ordered) - before
        dup_or_skipped = max(0, len(batch) - added)
        log(
            f"[batch {batch_idx}] +{added} new → {len(ordered)}/{target} unique "
            f"(parsed {len(batch)} lines; {dup_or_skipped} not added — duplicate or already had)"
        )
        if added == 0:
            print(
                "Warning: no new unique names this batch; model may be repeating. Retrying after delay.",
                file=sys.stderr,
                flush=True,
            )
        time.sleep(0.5)

    log(f"Done: {len(ordered)} unique first names collected.")
    return ordered[:target]


def mock_names(target: int, seed: int = 42, *, verbose: bool = True) -> list[str]:
    rng = random.Random(seed)
    first = [
        "Aarav",
        "Aditya",
        "Ananya",
        "Arjun",
        "Diya",
        "Ishaan",
        "Kavya",
        "Krishna",
        "Meera",
        "Neha",
        "Priya",
        "Rahul",
        "Riya",
        "Rohan",
        "Saanvi",
        "Vikram",
        "Vihaan",
        "Anika",
        "Dev",
        "Ira",
        "Kabir",
        "Lakshmi",
        "Manish",
        "Nisha",
        "Pooja",
        "Raj",
        "Sneha",
        "Suresh",
        "Tanvi",
        "Yash",
        "Aisha",
        "Vivaan",
        "Myra",
        "Reyansh",
        "Anvi",
        "Shaurya",
        "Kiara",
        "Advik",
        "Pari",
        "Dhruv",
        "Ishita",
        "Karan",
        "Navya",
        "Om",
        "Rudra",
        "Tara",
        "Ved",
        "Zara",
        "Harsh",
        "Jiya",
        "Kiran",
        "Lavanya",
        "Mira",
        "Nakul",
        "Ojas",
        "Pranav",
        "Rhea",
        "Samaira",
        "Tanya",
        "Uday",
        "Varun",
        "Yami",
        "Zayn",
    ]
    seen: set[str] = set()
    names: list[str] = []
    uid = 0
    step = max(1, min(250, target // 4))
    if verbose:
        print(f"[mock] generating {target} unique first names (progress every ~{step})…", flush=True)
    while len(names) < target:
        if verbose and names and len(names) % step == 0:
            print(f"[mock] {len(names)}/{target} unique…", flush=True)
        base = rng.choice(first)
        variant = base
        if rng.random() < 0.35:
            variant = base + rng.choice(["a", "i", "ya", "an", "vi", "u"])
        k = variant.lower()
        while k in seen:
            uid += 1
            variant = f"{base}{uid}"
            k = variant.lower()
        seen.add(k)
        names.append(variant[0].upper() + variant[1:] if len(variant) > 1 else variant.upper())
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate TrainingNames.txt (first names only, unique).")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="Output file path")
    ap.add_argument("-n", "--n", type=int, default=1000, help="Target unique first names")
    ap.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH, help="Names requested per Gemini call")
    ap.add_argument("--mock", action="store_true", help="Offline synthetic first names (no Gemini)")
    ap.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model id (when not using --mock)",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="I print less (no per-batch progress for Gemini).",
    )
    ap.add_argument(
        "--timeout-sec",
        type=int,
        default=180,
        help="HTTP timeout per Gemini request (default 180). Stops infinite hangs.",
    )
    ap.add_argument(
        "--heartbeat-sec",
        type=int,
        default=30,
        help="Print 'still waiting' every N seconds while the API call runs.",
    )
    args = ap.parse_args()
    verbose = not args.quiet

    if args.mock:
        names = mock_names(args.n, verbose=verbose)
    else:
        _load_dotenv()
        try:
            names = fetch_gemini(
                args.n,
                args.model,
                args.batch_size,
                verbose=verbose,
                timeout_ms=max(5_000, args.timeout_sec * 1000),
                heartbeat_sec=max(5, args.heartbeat_sec),
            )
        except Exception as e:
            print(f"Gemini failed ({e}); falling back to --mock.", file=sys.stderr)
            names = mock_names(args.n, verbose=verbose)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Wrote {len(names)} unique first names to {args.out}")


if __name__ == "__main__":
    main()
