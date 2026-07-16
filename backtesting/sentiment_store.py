"""As-of-dated sentiment reader for the backtest (strict point-in-time, fully local).

The harness never scrapes news in-loop (locked decision: sentiment is read-only).
It reads pre-collected, as-of-dated snapshots for the window ``[C - window, C]``
from **local files only** — the news Excel workbook's ``Sentiment`` sheet
(preferred) or a parquet cache — and returns them in the exact production shape
``{entity_id: [snapshot, ...]}`` that :func:`sentiment_context_for` consumes. There
is **no Firestore / Google-Sheet path**: everything the simulation needs lives on
disk.

Point-in-time guard: only snapshots with ``as_of_date <= C`` (and ``>= C-window``)
are returned. A window with no snapshot for an entity ⇒ that entity is simply
**absent** from the map, so ``sentiment_context_for`` yields all-``None`` and
``ask_glm`` receives ``"unavailable"`` — never a zero-fill that would look like
genuine neutral sentiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ingestion.aliases import sector_for
from ingestion.sectors import sector_entity_id

GENERAL_ENTITY = "GENERAL"
_SNAPSHOT_FIELDS = (
    "source", "as_of_date", "sent_mean_3d", "sent_mean_7d",
    "sent_pos_share", "sent_neg_share", "n_3d", "n_7d",
)


def _entities_for(tickers: List[str]) -> Dict[str, str]:
    """Map every relevant entity_id → its 'kind' (company/sector/general).

    A stock is allowed to see its own entity, its sector's entity, and GENERAL —
    matching the production ``sentiment_context_for`` visibility rules.
    """
    entities: Dict[str, str] = {GENERAL_ENTITY: "general"}
    for t in tickers:
        entities[t.upper()] = "company"
        sector = sector_for(t)
        if sector and sector != "Unknown":
            entities[sector_entity_id(sector)] = "sector"
    return entities


def window_asof(
    tickers: List[str],
    cutoff: Any,
    window_days: int = 7,
    *,
    workbook_path: Optional[Path] = None,
    cache_path: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return ``{entity_id: [snapshot]}`` for ``[cutoff-window_days, cutoff]``.

    Reads the local news workbook's ``Sentiment`` sheet when present, else a
    parquet cache. Both are on-disk; no network source is consulted.
    """
    C = pd.Timestamp(cutoff).normalize()
    lo = (C - pd.Timedelta(days=int(window_days))).normalize()
    wanted = _entities_for(tickers)

    if workbook_path is not None and Path(workbook_path).exists():
        rows = _from_excel(Path(workbook_path), wanted, lo, C)
    elif cache_path is not None and Path(cache_path).exists():
        rows = _from_cache(Path(cache_path), wanted, lo, C)
    else:
        return {}

    out: Dict[str, List[Dict[str, Any]]] = {}
    for entity_id, snap in rows:
        out.setdefault(entity_id, []).append(snap)
    return out


def _snapshot(entity_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    snap = {k: data.get(k) for k in _SNAPSHOT_FIELDS}
    # ``ticker`` in the production snapshot is really the entity id.
    snap.setdefault("source", data.get("source"))
    return snap


def _from_cache(path: Path, wanted: Dict[str, str], lo: pd.Timestamp, hi: pd.Timestamp):
    df = pd.read_parquet(path)
    if df.empty:
        return []
    ent_col = "entity_id" if "entity_id" in df.columns else ("ticker" if "ticker" in df.columns else None)
    if ent_col is None or "as_of_date" not in df.columns:
        return []
    dates = pd.to_datetime(df["as_of_date"], errors="coerce").dt.normalize()
    mask = (dates >= lo) & (dates <= hi) & df[ent_col].astype(str).str.upper().isin(
        {e.upper() for e in wanted}
    )
    result = []
    for _, row in df.loc[mask].iterrows():
        entity_id = str(row[ent_col]).upper()
        result.append((entity_id, _snapshot(entity_id, row.to_dict())))
    return result


SENTIMENT_SHEET = "Sentiment"


def _from_excel(path: Path, wanted: Dict[str, str], lo: pd.Timestamp, hi: pd.Timestamp):
    """Read as-of-dated snapshots from the news workbook's ``Sentiment`` sheet.

    Yields the same ``(entity_id, snapshot)`` tuples as :func:`_from_cache`, so the
    Excel and parquet paths are interchangeable. A workbook missing the sheet (or
    the required columns) simply yields nothing → entities read as "unavailable".
    """
    try:
        df = pd.read_excel(path, sheet_name=SENTIMENT_SHEET)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    ent_col = "entity_id" if "entity_id" in df.columns else ("ticker" if "ticker" in df.columns else None)
    if ent_col is None or "as_of_date" not in df.columns:
        return []
    dates = pd.to_datetime(df["as_of_date"], errors="coerce").dt.normalize()
    mask = (dates >= lo) & (dates <= hi) & df[ent_col].astype(str).str.upper().isin(
        {e.upper() for e in wanted}
    )
    result = []
    for _, row in df.loc[mask].iterrows():
        entity_id = str(row[ent_col]).upper()
        result.append((entity_id, _snapshot(entity_id, row.to_dict())))
    return result


def sent_fingerprint(sent_map: Dict[str, List[Dict[str, Any]]], ticker: str) -> str:
    """7-day change-detection fingerprint of a ticker's *visible* sentiment.

    Includes the company, its sector and GENERAL entities so a change in any of
    the three that the stock can see triggers a re-suggestion on the sentiment
    cadence.
    """
    sector = sector_for(ticker)
    sector_ent = sector_entity_id(sector) if sector and sector != "Unknown" else None
    relevant = {
        "company": sent_map.get(ticker.upper()),
        "sector": sent_map.get(sector_ent.upper()) if sector_ent else None,
        "general": sent_map.get(GENERAL_ENTITY),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()
