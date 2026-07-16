"""Historical fundamentals collector for the backtest — source: screener.in.

Why screener.in (not yfinance): yfinance only exposes the last ~4-5 quarters, so
it cannot supply point-in-time fundamentals for a 2020-2026 backtest. screener.in
publishes, on each company's public page, a **quarterly results** table (~12
quarters) *and* a 10-year **annual** P&L series — enough depth to give the LLM
real fundamentals across the whole window.

Point-in-time discipline (the whole reason this exists):
    Each period is stamped with an **information-availability date** — NOT today —
    written to the ``scrape_date`` column the simulation gates on
    (``load_pit_asof("fundamentals", C, "scrape_date")`` keeps only rows with
    ``scrape_date <= C``). Availability = period-end + a reporting lag
    (quarterly ~45d, annual ~60d), so a quarter that had not yet been announced by
    a past cutoff ``C`` is correctly invisible at ``C``. This mirrors the yfinance
    fallback gate in :func:`backtesting.signals.fundamentals_asof`.

Output: append-only rows to the fundamentals PIT parquet (default
``ingestion._pit.pit_path("fundamentals")`` — the exact file the harness reads),
with columns ``ticker, quarter, quarter_end_date, scrape_date, period_type,
source, financials_json``. Re-runs are idempotent (dedup on
``ticker, quarter, scrape_date``); new tickers / periods simply append.

Usage::

    python -m backtesting.fetch_fundamentals                       # backtest universe
    python -m backtesting.fetch_fundamentals --tickers TCS,RELIANCE
    python -m backtesting.fetch_fundamentals --dry-run             # parse, don't write
    python -m backtesting.fetch_fundamentals --no-annual           # quarterly only

If screener.in blocks anonymous requests (HTTP 403/429), set a logged-in cookie::

    export SCREENER_COOKIE="sessionid=...; csrftoken=..."
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.config.backtest_settings import BacktestSettings, get_backtest_settings

SCREENER_BASE = "https://www.screener.in/company"
QUARTERS_SECTION = "quarters"
ANNUAL_SECTION = "profit-loss"

# screener row-label (normalized) → our financials key.
_METRIC_MAP = {
    "sales": "revenue",
    "revenue": "revenue",
    "expenses": "expenses",
    "operating profit": "operating_profit",
    "opm %": "operating_margin_pct",
    "other income": "other_income",
    "interest": "interest",
    "depreciation": "depreciation",
    "profit before tax": "profit_before_tax",
    "tax %": "tax_pct",
    "net profit": "net_income",
    "eps in rs": "eps",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Pure parsing helpers (unit-testable; no network)
# ---------------------------------------------------------------------------

def _num(raw: Any) -> Optional[float]:
    """Parse a screener cell (``'12,345'``, ``'23%'``, ``'-'``, ``''``) → float|None."""
    if raw is None:
        return None
    s = str(raw).strip().replace(",", "").replace("%", "").replace("₹", "")
    if s in ("", "-", "—", "nan", "NaN"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_label(label: Any) -> str:
    return (
        str(label).replace("\xa0", " ").replace("+", " ").strip().lower()
    )


def _period_end(label: str) -> Optional[pd.Timestamp]:
    """``'Jun 2022'`` → 2022-06-30 (last day of the period-end month)."""
    try:
        ts = pd.to_datetime(str(label).strip(), format="%b %Y")
    except (ValueError, TypeError):
        try:
            ts = pd.to_datetime(str(label).strip())
        except (ValueError, TypeError):
            return None
    return (ts + pd.offsets.MonthEnd(0)).normalize()


def _quarter_key(end: pd.Timestamp) -> str:
    """Calendar-quarter key, e.g. 2022-06-30 → ``'2022Q2'``."""
    return f"{end.year}Q{(end.month - 1) // 3 + 1}"


def _fy_key(end: pd.Timestamp) -> str:
    """Indian fiscal-year key for an annual (Mar-ending) column → ``'FY2022'``.

    A March 2022 close is FY2021-22; label it FY2022. Non-March ends fall back to
    the calendar year so nothing is dropped.
    """
    return f"FY{end.year}" if end.month == 3 else f"FY{end.year}"


def parse_period_table(table_html: str) -> Dict[str, Dict[str, Any]]:
    """Parse one screener period table → ``{period_label: {metric_key: value}}``.

    ``table_html`` is the ``<table>`` of the quarterly-results or annual-P&L
    section. Columns are period labels (``'Jun 2022'`` / ``'Mar 2020'``); the
    first column holds the metric names. Only mapped metrics are kept.
    """
    from io import StringIO
    frames = pd.read_html(StringIO(table_html))
    if not frames:
        return {}
    df = frames[0]
    metric_col = df.columns[0]
    out: Dict[str, Dict[str, Any]] = {}
    for period in df.columns[1:]:
        label = str(period).strip()
        if _period_end(label) is None:  # skip 'TTM' and other non-date columns
            continue
        col: Dict[str, Any] = {}
        for _, row in df.iterrows():
            key = _METRIC_MAP.get(_normalize_label(row[metric_col]))
            if key is not None:
                col[key] = _num(row[period])
        if col:
            out[label] = col
    return out


def _annual_availability(end: pd.Timestamp, refresh_month: int) -> pd.Timestamp:
    """When an annual result becomes usable: the ``refresh_month`` on/after period-end.

    Indian annuals are audited and published a few months after the March close, so
    the backtest treats FY ending Mar YYYY as visible from **June YYYY** (default
    ``refresh_month=6``). A Jan/Feb 2020 cutoff therefore still sees FY2019; a
    June/July 2020 cutoff sees FY2020. Non-March ends roll to the next refresh month
    after the close so availability is never before the period ended.
    """
    avail_year = end.year if end.month <= refresh_month else end.year + 1
    return pd.Timestamp(year=avail_year, month=refresh_month, day=1)


def to_pit_records(
    ticker: str,
    parsed: Dict[str, Dict[str, Any]],
    *,
    period_type: str,
    lag_days: int,
    annual_refresh_month: int = 6,
) -> List[Dict[str, Any]]:
    """Turn parsed ``{label: financials}`` into availability-dated PIT rows.

    Availability (``scrape_date`` — the column the sim gates on) differs by type:
    quarterly = ``quarter_end + lag_days`` (~45d, results-announcement lag); annual
    = the ``annual_refresh_month`` (June) on/after the fiscal-year close.
    """
    records: List[Dict[str, Any]] = []
    for label, financials in parsed.items():
        end = _period_end(label)
        if end is None:
            continue
        if period_type == "annual":
            avail = _annual_availability(end, annual_refresh_month)
            key = _fy_key(end)
        else:
            avail = (end + pd.Timedelta(days=int(lag_days))).normalize()
            key = _quarter_key(end)
        records.append({
            "ticker": ticker.upper(),
            "quarter": key,
            "quarter_end_date": end.date().isoformat(),
            "scrape_date": avail.date().isoformat(),   # PIT availability date
            "period_type": period_type,
            "source": "screener.in",
            "financials_json": json.dumps(financials, default=str),
        })
    return records


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def screener_urls(symbol: str) -> List[str]:
    """Candidate screener URLs (consolidated preferred, standalone fallback)."""
    from urllib.parse import quote
    enc = quote(symbol.upper(), safe="")
    return [f"{SCREENER_BASE}/{enc}/consolidated/", f"{SCREENER_BASE}/{enc}/"]


def fetch_company_html(symbol: str, session: Any, *, retries: int = 3, pause: float = 2.0) -> Optional[str]:
    """Fetch a company's screener page HTML (tries consolidated then standalone)."""
    for url in screener_urls(symbol):
        for attempt in range(1, retries + 1):
            try:
                resp = session.get(url, headers=_headers(), timeout=30)
                if resp.status_code == 200 and resp.text:
                    return resp.text
                if resp.status_code in (403, 429):
                    _log(f"[fund] {symbol}: HTTP {resp.status_code} (attempt {attempt}) — "
                         f"set SCREENER_COOKIE if this persists")
                    time.sleep(pause * attempt * 2)
                    continue
                break  # 404 etc. → try next URL
            except Exception as exc:  # noqa: BLE001 — retried
                _log(f"[fund] {symbol}: fetch error {exc} (attempt {attempt})")
                time.sleep(pause * attempt)
    return None


def _headers() -> Dict[str, str]:
    h = dict(_HEADERS)
    cookie = os.environ.get("SCREENER_COOKIE")
    if cookie:
        h["Cookie"] = cookie
    return h


def _section_table_html(html: str, section_id: str) -> Optional[str]:
    """Return the ``<table>`` HTML inside ``<section id=section_id>``."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    section = soup.find("section", id=section_id) or soup.find(id=section_id)
    if section is None:
        return None
    table = section.find("table")
    return str(table) if table is not None else None


def collect_symbol(
    symbol: str,
    session: Any,
    *,
    quarterly_lag: int,
    annual_refresh_month: int,
    include_annual: bool,
    pause: float,
) -> List[Dict[str, Any]]:
    """Fetch + parse one company → PIT records (quarterly, plus annual if enabled)."""
    html = fetch_company_html(symbol, session, pause=pause)
    if not html:
        _log(f"[fund] {symbol}: no page fetched — skipped")
        return []
    records: List[Dict[str, Any]] = []

    q_html = _section_table_html(html, QUARTERS_SECTION)
    if q_html:
        records += to_pit_records(symbol, parse_period_table(q_html),
                                  period_type="quarterly", lag_days=quarterly_lag)
    if include_annual:
        a_html = _section_table_html(html, ANNUAL_SECTION)
        if a_html:
            records += to_pit_records(symbol, parse_period_table(a_html),
                                      period_type="annual", lag_days=quarterly_lag,
                                      annual_refresh_month=annual_refresh_month)
    _log(f"[fund] {symbol}: {len(records)} period records "
         f"({'q+a' if include_annual else 'q'})")
    return records


# ---------------------------------------------------------------------------
# Storage (append-only PIT parquet the harness reads)
# ---------------------------------------------------------------------------

def _pit_path(override: Optional[str]) -> Path:
    if override:
        return Path(override)
    from ingestion._pit import pit_path
    return pit_path("fundamentals")


def store_records(records: List[Dict[str, Any]], *, pit_path: Path) -> int:
    """Append records to the fundamentals PIT parquet; dedup keeps re-runs idempotent."""
    if not records:
        return 0
    from ingestion._pit import append_pit
    df = pd.DataFrame(records)
    return append_pit(df, "fundamentals",
                      dedup_keys=["ticker", "quarter", "scrape_date"],
                      path=pit_path, log=_log)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def collect(
    tickers: List[str],
    *,
    settings: Optional[BacktestSettings] = None,
    include_annual: bool = True,
    quarterly_lag: Optional[int] = None,
    annual_refresh_month: int = 6,
    pause: float = 2.5,
    dry_run: bool = False,
    pit_path_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect screener fundamentals for ``tickers`` into the PIT store."""
    import requests
    settings = settings or get_backtest_settings()
    quarterly_lag = settings.reporting_lag_days if quarterly_lag is None else quarterly_lag
    session = requests.Session()

    all_records: List[Dict[str, Any]] = []
    for i, sym in enumerate(tickers, 1):
        recs = collect_symbol(sym, session, quarterly_lag=quarterly_lag,
                              annual_refresh_month=annual_refresh_month,
                              include_annual=include_annual, pause=pause)
        all_records += recs
        if i < len(tickers):
            time.sleep(pause)   # be polite to screener.in

    written = 0
    path = _pit_path(pit_path_override)
    if dry_run:
        for r in all_records[:12]:
            _log(f"[fund][dry] {r['ticker']} {r['quarter']} end={r['quarter_end_date']} "
                 f"avail={r['scrape_date']} {r['financials_json'][:80]}")
    else:
        written = store_records(all_records, pit_path=path)

    summary = {
        "status": "ok",
        "tickers": len(tickers),
        "records_parsed": len(all_records),
        "rows_after_merge": written,
        "pit_path": str(path),
        "dry_run": dry_run,
    }
    print(json.dumps(summary, default=str))
    return summary


def run(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Historical fundamentals collector (screener.in) → PIT store.")
    p.add_argument("--tickers", default=None, help="Comma-separated subset (default: backtest universe).")
    p.add_argument("--no-annual", action="store_true", help="Quarterly results only (skip the 10y annual series).")
    p.add_argument("--quarterly-lag-days", type=int, default=None,
                   help="Availability lag after a quarter-end (default: settings.reporting_lag_days=45).")
    p.add_argument("--annual-refresh-month", type=int, default=6,
                   help="Calendar month an annual result becomes usable (default: 6 = June).")
    p.add_argument("--pause", type=float, default=2.5, help="Seconds between company fetches (politeness).")
    p.add_argument("--pit-path", default=None, help="Override the fundamentals PIT parquet path.")
    p.add_argument("--dry-run", action="store_true", help="Parse and print; do not write the PIT store.")
    args = p.parse_args(argv)

    settings = get_backtest_settings()
    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else settings.resolved_tickers()
    )
    if not tickers:
        _log("[fund] no tickers resolved — aborting")
        return {"status": "no_tickers"}

    return collect(
        tickers, settings=settings, include_annual=not args.no_annual,
        quarterly_lag=args.quarterly_lag_days, annual_refresh_month=args.annual_refresh_month,
        pause=args.pause, dry_run=args.dry_run, pit_path_override=args.pit_path,
    )


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result.get("status") in ("ok", "no_tickers") else 1)
