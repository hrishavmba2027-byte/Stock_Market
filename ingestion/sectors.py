"""Sector detection + sector-entity helpers (Phase 3).

Production news is now collected per company AND per sector. A headline maps to
a sector when it either (a) names a covered company (the company's sector is
implied) or (b) matches one of the curated per-sector terms below — so genuine
sector-wide news ("IT services exporters rally") is captured even when no
specific company is named.

Sector sentiment is stored under a synthetic entity id (``sector_entity_id``)
so it lives alongside company docs in the same Firestore collection without
colliding with real tickers. ``ingestion.aliases.sector_for`` gives a company's
sector; both the ingest side (tagging) and the read side (the LLM layer) call
``sector_entity_id`` so the ids round-trip.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Pattern

from ingestion.aliases import list_tickers, load_aliases, sector_for

SECTOR_ENTITY_PREFIX = "SECTOR__"

# Curated, conservative sector terms. Kept tight to avoid false positives —
# company aliases already cover company-specific news; these catch sector-wide
# stories. Matching is word-boundary and case-insensitive.
SECTOR_TERMS: Dict[str, List[str]] = {
    "Auto": ["auto sector", "automobile", "automaker", "auto industry",
             "passenger vehicles", "two-wheeler", "auto sales", "auto stocks"],
    "Cement & Construction": ["cement sector", "cement industry", "cement prices",
                              "construction sector", "infrastructure sector", "realty sector"],
    "Chemicals": ["chemical sector", "chemicals industry", "specialty chemicals", "petrochemical"],
    "Consumer Durables": ["consumer durables", "white goods", "appliances sector"],
    "FMCG": ["FMCG", "fast-moving consumer goods", "consumer staples", "staples sector"],
    "Financials": ["banking sector", "financial sector", "NBFC", "PSU banks",
                   "private banks", "banking stocks", "lenders"],
    "Healthcare": ["pharma sector", "pharmaceutical", "healthcare sector",
                   "drugmakers", "pharma stocks"],
    "IT": ["IT sector", "information technology sector", "IT services",
           "software exporters", "IT stocks", "tech stocks"],
    "Metals & Mining": ["metals sector", "mining sector", "steel sector",
                        "metal stocks", "steelmakers"],
    "Oil & Gas": ["oil and gas", "oil & gas", "energy sector", "crude oil",
                  "refiners", "OMCs"],
    "Power": ["power sector", "power utilities", "power stocks", "electricity sector"],
    "Services": ["services sector"],
    "Telecom": ["telecom sector", "telecom industry", "telco", "telecom stocks", "spectrum auction"],
}


@lru_cache(maxsize=1)
def list_sectors() -> List[str]:
    """Sectors actually present in the ticker universe."""
    return sorted({v.get("sector") for v in load_aliases().values() if v.get("sector")})


def sector_entity_id(sector: str) -> str:
    """Deterministic synthetic entity id for a sector (e.g. ``SECTOR__OIL_GAS``)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", sector or "").strip("_").upper()
    return f"{SECTOR_ENTITY_PREFIX}{slug}"


def is_sector_entity(entity: str) -> bool:
    return str(entity or "").startswith(SECTOR_ENTITY_PREFIX)


@lru_cache(maxsize=1)
def _sector_id_to_name() -> Dict[str, str]:
    return {sector_entity_id(s): s for s in list_sectors()}


def sector_name_for_entity(entity: str) -> str:
    return _sector_id_to_name().get(entity, entity)


@lru_cache(maxsize=1)
def _sector_regex() -> Dict[str, Pattern[str]]:
    out: Dict[str, Pattern[str]] = {}
    for sector in list_sectors():
        terms = {t.strip() for t in SECTOR_TERMS.get(sector, []) if t and t.strip()}
        if not terms:
            continue
        escaped = sorted((re.escape(t) for t in terms), key=len, reverse=True)
        pattern = r"(?<![A-Za-z0-9])(?:" + "|".join(escaped) + r")(?![A-Za-z0-9])"
        out[sector] = re.compile(pattern, re.IGNORECASE)
    return out


def find_sectors_in_text(text: str) -> List[str]:
    """Sectors explicitly named in ``text`` via the curated term lists."""
    if not text:
        return []
    return [sector for sector, pat in _sector_regex().items() if pat.search(text)]


def sectors_for_companies(tickers: List[str]) -> List[str]:
    """Sectors implied by a list of matched company tickers."""
    seen: List[str] = []
    for t in tickers:
        sec = sector_for(t)
        if sec and sec != "Unknown" and sec not in seen:
            seen.append(sec)
    return seen
