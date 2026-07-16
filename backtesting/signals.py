"""Signal generation for the backtest — reuses the production LLM layer verbatim.

``build_market`` shapes the exact ``market`` dict production feeds ``ask_glm``;
``fundamentals_asof`` returns point-in-time fundamentals; ``run_signals`` calls
the *unchanged* production ``ask_glm`` per ticker (with the same incremental
fingerprint skip) so the decisions are byte-for-byte what production would make.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from features.trade_suggestions import (
    GLM_MODEL,
    FUNDAMENTALS_QUARTERS,
    RECENT_CLOSES,
    _has_sentiment,
    ask_glm,
    fingerprint,
    sentiment_context_for,
)

FORECAST_KEYS = [f"T+{h}" for h in range(1, 16)]


def build_market(
    sym: str,
    frame_asof: pd.DataFrame,
    forecast_path: Dict[str, float],
    cutoff: Any,
) -> Dict[str, Any]:
    """``{date, close, recent_closes, forecast}`` for a single stock at bar ``C``."""
    closes = pd.to_numeric(frame_asof.get("Close"), errors="coerce").dropna().to_numpy(dtype=float)
    close = float(closes[-1]) if len(closes) else None
    return {
        "date": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
        "close": close,
        "recent_closes": [float(x) for x in closes[-RECENT_CLOSES:]],
        "forecast": {k: float(v) for k, v in (forecast_path or {}).items()},
    }


def fundamentals_asof(
    tickers: List[str],
    cutoff: Any,
    *,
    reporting_lag_days: int = 45,
    use_yfinance_fallback: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Point-in-time fundamentals per ticker, compacted to the latest 3 quarters.

    Primary path: the append-only PIT archive filtered on ``scrape_date <= C``
    (information-availability date). Fallback (only if enabled): a live yfinance
    pull gated by ``quarter_end_date + reporting_lag_days <= C`` so a results lag
    can't leak an as-yet-unannounced quarter.
    """
    C = pd.Timestamp(cutoff).normalize()
    out: Dict[str, Dict[str, Any]] = {}
    pit = _load_pit_fundamentals(C)
    for t in tickers:
        rows = pit.get(t.upper())
        if rows:
            out[t.upper()] = _compact(rows)
    missing = [t for t in tickers if t.upper() not in out]
    if missing and use_yfinance_fallback:
        for t in missing:
            compact = _yfinance_fallback(t, C, reporting_lag_days)
            if compact:
                out[t.upper()] = compact
    return out


def _load_pit_fundamentals(C: pd.Timestamp) -> Dict[str, List[Dict[str, Any]]]:
    try:
        from ingestion._pit import load_pit_asof
    except Exception:  # pragma: no cover
        return {}
    try:
        df = load_pit_asof("fundamentals", C, "scrape_date")
    except Exception:  # pragma: no cover
        return {}
    if df is None or df.empty or "ticker" not in df.columns:
        return {}
    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in df.iterrows():
        d = row.to_dict()
        fin = d.get("financials")
        if fin is None and d.get("financials_json"):
            try:
                fin = json.loads(d["financials_json"])
            except Exception:
                fin = {}
        by_ticker.setdefault(str(d.get("ticker")).upper(), []).append({
            "quarter": d.get("quarter"),
            "quarter_end_date": d.get("quarter_end_date"),
            "scrape_date": d.get("scrape_date"),
            "period_type": d.get("period_type"),
            "financials": fin or {},
        })
    return by_ticker


def _compact(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Latest ``FUNDAMENTALS_QUARTERS`` (5) periods, quarterly preferred over annual.

    Quarterly results are used whenever any are visible at the cutoff; only when no
    quarterly data exists (e.g. early backtest years screener lacks quarterly for)
    do we fall back to the annual series — never mixing the two. Rows without a
    ``period_type`` (production yfinance records) are treated as quarterly. The
    result is a rolling window of up to five periods, newest first.
    """
    def _is_annual(r: Dict[str, Any]) -> bool:
        return str(r.get("period_type") or "quarterly").lower() == "annual"

    quarterly = [r for r in rows if not _is_annual(r)]
    chosen = quarterly if quarterly else rows
    ordered = sorted(
        chosen, key=lambda r: (str(r.get("quarter_end_date") or ""), str(r.get("quarter") or "")),
        reverse=True,
    )[:FUNDAMENTALS_QUARTERS]
    recent = {
        str(r.get("quarter")): {
            "quarter_end_date": r.get("quarter_end_date"),
            "period_type": r.get("period_type") or "quarterly",
            "financials": r.get("financials") or {},
        }
        for r in ordered
    }
    scrapes = [str(r.get("scrape_date")) for r in ordered if r.get("scrape_date")]
    return {
        "scrape_date": max(scrapes) if scrapes else None,
        "latest_quarter": ordered[0].get("quarter") if ordered else None,
        "recent_quarters": recent,
    }


def _yfinance_fallback(ticker: str, C: pd.Timestamp, lag_days: int) -> Optional[Dict[str, Any]]:
    try:
        from ingestion.fundamentals import fetch_quarterly_records
        records = fetch_quarterly_records(ticker, today=C.date())
    except Exception:  # pragma: no cover - network/parse errors are non-fatal
        return None
    gate = C - pd.Timedelta(days=int(lag_days))
    visible = []
    for rec in records or []:
        qend = pd.to_datetime(rec.get("quarter_end_date"), errors="coerce")
        if pd.notna(qend) and qend.normalize() <= gate:
            visible.append(rec)
    return _compact(visible) if visible else None


def run_signals(
    tickers: List[str],
    forecasts: Dict[str, Dict[str, float]],
    frames_asof: Dict[str, pd.DataFrame],
    sent_map: Dict[str, List[Dict[str, Any]]],
    funds: Dict[str, Dict[str, Any]],
    cutoff: Any,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    prev_fingerprints: Optional[Dict[str, str]] = None,
    workers: int = 4,
    ask: Callable[..., Dict[str, Any]] = ask_glm,
) -> Dict[str, Dict[str, Any]]:
    """Per-ticker BUY/HOLD/AVOID via ``ask`` (defaults to production ``ask_glm``).

    Skips a ticker whose fingerprint (forecast + sentiment + fundamentals) is
    unchanged from ``prev_fingerprints`` — the same "contextualize once, then only
    new data" contract production uses. ``ask`` is injectable so tests run without
    the network.
    """
    model = model or os.environ.get("GLM_MODEL", GLM_MODEL)
    api_key = api_key or os.environ.get("GLM_API_KEY", "")
    prev_fingerprints = prev_fingerprints or {}

    jobs = []
    for t in tickers:
        fc = forecasts.get(t)
        frame = frames_asof.get(t)
        if not fc or frame is None or frame.empty:
            continue
        market = build_market(t, frame, fc, cutoff)
        snap = sentiment_context_for(t, sent_map)
        fund = funds.get(t.upper())
        fp = fingerprint({"market": market, "sentiment": snap, "fundamentals": fund})
        if prev_fingerprints.get(t) == fp:
            continue
        jobs.append((t, market, snap, fund, fp))

    results: Dict[str, Dict[str, Any]] = {}

    def _one(job):
        t, market, snap, fund, fp = job
        suggestion = ask(t, market, snap, model, api_key, fund)
        return t, {
            "as_of_date": market["date"],
            "last_close": market["close"],
            "action": suggestion.get("action"),
            "buy_price": suggestion.get("buy_price"),
            "sell_day": suggestion.get("sell_day"),
            "sell_price": suggestion.get("sell_price"),
            "stoploss": suggestion.get("stoploss"),
            "confidence": suggestion.get("confidence"),
            "rationale": suggestion.get("rationale"),
            "sentiment_used": _has_sentiment(snap),
            "fundamentals_used": bool(fund),
            "fingerprint": fp,
        }

    if workers <= 1:
        for job in jobs:
            try:
                t, sig = _one(job)
                results[t] = sig
            except Exception as exc:  # noqa: BLE001 - per-stock isolation
                results[job[0]] = {"error": str(exc), "fingerprint": job[4]}
        return results

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_one, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                t, sig = fut.result()
                results[t] = sig
            except Exception as exc:  # noqa: BLE001
                results[job[0]] = {"error": str(exc), "fingerprint": job[4]}
    return results
