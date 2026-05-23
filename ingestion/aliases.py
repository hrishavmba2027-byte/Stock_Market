from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Pattern

ALIASES_PATH = Path(__file__).resolve().parent / "ticker_aliases.json"


@lru_cache(maxsize=1)
def load_aliases() -> Dict[str, Dict]:
    with ALIASES_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def list_tickers() -> List[str]:
    return sorted(load_aliases().keys())


def sector_for(ticker: str) -> str:
    return load_aliases().get(ticker.upper(), {}).get("sector", "Unknown")


def all_aliases_for(ticker: str) -> List[str]:
    entry = load_aliases().get(ticker.upper())
    if not entry:
        return [ticker]
    return [ticker, entry["name"], *entry.get("aliases", [])]


@lru_cache(maxsize=1)
def _alias_regex_per_ticker() -> Dict[str, Pattern[str]]:
    out: Dict[str, Pattern[str]] = {}
    for ticker in list_tickers():
        terms = {term.strip() for term in all_aliases_for(ticker) if term and term.strip()}
        escaped = sorted((re.escape(t) for t in terms), key=len, reverse=True)
        pattern = r"(?<![A-Za-z0-9])(?:" + "|".join(escaped) + r")(?![A-Za-z0-9])"
        out[ticker] = re.compile(pattern, re.IGNORECASE)
    return out


def find_tickers_in_text(text: str) -> List[str]:
    if not text:
        return []
    hits: List[str] = []
    for ticker, pattern in _alias_regex_per_ticker().items():
        if pattern.search(text):
            hits.append(ticker)
    return hits
