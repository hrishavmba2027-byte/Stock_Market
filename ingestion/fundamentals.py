"""P1.1 — Quarterly fundamentals & earnings event features.

Pulls per-ticker financials from yfinance and stages quarterly records for
Firestore upload:

1. **Company Firestore documents** (collection ``fundamentals``) — one
   document per ticker containing the last 4 quarters:

        {
            "company_name": "Reliance Industries Limited",
            "scrape_date":  "2026-05-20",
            "ticker":       "RELIANCE",
            "quarters": {
                "2025Q3": {
                    "quarter_end_date": "2025-09-30",
                    "financials": { revenue, net_income, ... }
                }
            }
        }

   Document id = ``{TICKER}`` (e.g. ``RELIANCE``). Re-running the job
   overwrites the existing company doc idempotently. Successful
   uploads delete the staged parquet; failed/skipped uploads retain it under
   ``Data/archive`` for retry.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time as time_module
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ingestion._archive import (
    archive_path_for,
    drain_pending_archive,
    merge_stage_to_archive,
    truthy_env,
    upload_and_clear_archive,
)
from ingestion._firestore import init_firestore_client
from ingestion.aliases import list_tickers, load_aliases, sector_for

# Quarterly records destined for Firestore (one row per (ticker, quarter) with
# a JSON-encoded ``financials_json`` column). This is a temporary staging file:
# successful Firestore uploads delete it; failed/skipped uploads retain it.
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "Data" / "archive" / "fundamentals.parquet"
FLAT_SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "Data" / "archive" / "fundamentals_flat.parquet"
DEFAULT_LOOKBACK_QUARTERS = 4
DEFAULT_EVENT_WINDOW_DAYS = 5
FIRESTORE_COLLECTION = "fundamentals"
FIRESTORE_BATCH_SIZE = 200  # well under the 500-op hard limit
NSE_SUFFIX = ".NS"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _yf_ticker(symbol: str):
    import yfinance as yf

    return yf.Ticker(f"{symbol}{NSE_SUFFIX}")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        if isinstance(value, (pd.Series, pd.DataFrame)):
            return None
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _row_value(df: Optional[pd.DataFrame], key: str, col: int = 0) -> Optional[float]:
    """Pick a numeric cell out of a yfinance financial statement frame."""
    if df is None or df.empty:
        return None
    if key not in df.index:
        return None
    row = df.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    if not isinstance(row, pd.Series) or row.empty:
        return None
    if col >= len(row):
        return None
    return _safe_float(row.iloc[col])


def _yoy(current: Optional[float], year_ago: Optional[float]) -> Optional[float]:
    if current is None or year_ago is None or year_ago == 0:
        return None
    return (current - year_ago) / abs(year_ago)


def _ratio(num: Optional[float], denom: Optional[float]) -> Optional[float]:
    if num is None or denom is None or denom == 0:
        return None
    return num / denom


def _next_earnings_date(ticker: Any, today: date) -> Optional[date]:
    """yfinance returns past + upcoming earnings; find the next one strictly in the future."""
    try:
        df = ticker.earnings_dates
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    try:
        index = pd.to_datetime(df.index, utc=True, errors="coerce")
    except Exception:
        return None
    upcoming: List[date] = []
    for ts in index:
        if pd.isna(ts):
            continue
        d = ts.date()
        if d > today:
            upcoming.append(d)
    if not upcoming:
        return None
    return min(upcoming)


def _quarter_label(d: Any) -> Optional[str]:
    """Convert a yfinance column header (date / Timestamp / str) to ``YYYYQn``.

    yfinance uses the period-end date as the column key, so a date in
    September → Q3 of that year, December → Q4, March → Q1, June → Q2.
    """
    try:
        ts = pd.to_datetime(d)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    quarter = ((int(ts.month) - 1) // 3) + 1
    return f"{int(ts.year)}Q{quarter}"


def _quarter_columns(*frames: Optional[pd.DataFrame]) -> List[Tuple[int, pd.Timestamp]]:
    """Return ``[(col_idx, timestamp), …]`` sorted most-recent first.

    Driven by whichever of the supplied frames has the most usable columns
    so that a missing income statement doesn't kill the balance-sheet path.
    """
    best: List[Tuple[int, pd.Timestamp]] = []
    for df in frames:
        if df is None or df.empty:
            continue
        parsed: List[Tuple[int, pd.Timestamp]] = []
        for idx, col in enumerate(df.columns):
            try:
                ts = pd.to_datetime(col)
            except Exception:
                continue
            if pd.isna(ts):
                continue
            parsed.append((idx, pd.Timestamp(ts)))
        if len(parsed) > len(best):
            best = parsed
    best.sort(key=lambda x: x[1], reverse=True)
    return best


def _company_name(ticker: Any, symbol: str) -> str:
    """Curated name from the alias map, falling back to yfinance ``.info``."""
    try:
        entry = load_aliases().get(symbol.upper(), {})
        if entry.get("name"):
            return entry["name"]
    except Exception:
        pass
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    return info.get("longName") or info.get("shortName") or symbol


def _extract_quarter_financials(
    income: Optional[pd.DataFrame],
    balance: Optional[pd.DataFrame],
    cashflow: Optional[pd.DataFrame],
    col_idx: int,
) -> Dict[str, Any]:
    """Pull every metric at column position ``col_idx`` into a flat dict.

    Numbers are scoped to a single quarter (yfinance ``quarterly_*``).
    Ratios are computed on those quarter values; comparing across quarters
    is the consumer's job.
    """
    revenue = _row_value(income, "Total Revenue", col_idx)
    net_income = _row_value(income, "Net Income", col_idx)
    operating_income = _row_value(income, "Operating Income", col_idx)
    gross_profit = _row_value(income, "Gross Profit", col_idx)

    total_equity = _row_value(balance, "Total Stockholder Equity", col_idx) or _row_value(
        balance, "Stockholders Equity", col_idx
    )
    total_debt = _row_value(balance, "Total Debt", col_idx)
    total_assets = _row_value(balance, "Total Assets", col_idx)

    op_cashflow = _row_value(cashflow, "Total Cash From Operating Activities", col_idx) or _row_value(
        cashflow, "Operating Cash Flow", col_idx
    )
    capex = _row_value(cashflow, "Capital Expenditures", col_idx) or _row_value(
        cashflow, "Capital Expenditure", col_idx
    )
    fcf = None
    if op_cashflow is not None and capex is not None:
        fcf = op_cashflow + capex  # capex is negative in yfinance convention

    return {
        # Income statement
        "revenue": revenue,
        "net_income": net_income,
        "operating_income": operating_income,
        "gross_profit": gross_profit,
        "operating_margin": _ratio(operating_income, revenue),
        "net_margin": _ratio(net_income, revenue),
        "gross_margin": _ratio(gross_profit, revenue),
        # Balance sheet
        "total_equity": total_equity,
        "total_debt": total_debt,
        "total_assets": total_assets,
        "debt_to_equity": _ratio(total_debt, total_equity),
        "roe": _ratio(net_income, total_equity),
        "asset_turnover": _ratio(revenue, total_assets),
        # Cash flow
        "operating_cashflow": op_cashflow,
        "capex": capex,
        "free_cash_flow": fcf,
    }


def fetch_quarterly_records(
    symbol: str,
    today: Optional[date] = None,
    lookback: int = DEFAULT_LOOKBACK_QUARTERS,
) -> List[Dict[str, Any]]:
    """Return the last ``lookback`` quarters as Firestore-ready records.

    Each record has the schema:
        {company_name, scrape_date, quarter, financials, ticker, quarter_end_date}

    ``ticker`` and ``quarter_end_date`` are included for query convenience;
    the four user-specified fields (``company_name``, ``scrape_date``,
    ``quarter``, ``financials``) are always present.
    """
    today = today or date.today()
    ticker = _yf_ticker(symbol)
    company_name = _company_name(ticker, symbol)

    try:
        income = ticker.quarterly_financials
    except Exception:
        income = None
    try:
        balance = ticker.quarterly_balance_sheet
    except Exception:
        balance = None
    try:
        cashflow = ticker.quarterly_cashflow
    except Exception:
        cashflow = None

    columns = _quarter_columns(income, balance, cashflow)
    if not columns:
        return []

    records: List[Dict[str, Any]] = []
    k = 0
    for col_idx, col_ts in columns[:lookback]:
        quarter = _quarter_label(col_ts)
        if quarter is None:
            continue
        records.append(
            {
                "ticker": symbol,
                "company_name": company_name,
                "scrape_date": today.isoformat(),
                "quarter": quarter,
                "quarter_end_date": col_ts.date().isoformat(),
                "financials": _extract_quarter_financials(income, balance, cashflow, col_idx),
            }
        )
    return records


def fetch_fundamentals(symbol: str, today: Optional[date] = None) -> Dict[str, Any]:
    """Return a flat dict of fundamental features for one ticker (ML-pipeline row).

    Kept for backward compatibility with the parquet feature pipeline.
    The richer per-quarter view lives in :func:`fetch_quarterly_records`.
    """
    today = today or date.today()
    ticker = _yf_ticker(symbol)

    try:
        info = ticker.info or {}
    except Exception as exc:  # yfinance occasionally raises on .info
        _log(f"[fundamentals] info failed for {symbol}: {exc}")
        info = {}

    try:
        income = ticker.quarterly_financials
    except Exception:
        income = None
    try:
        balance = ticker.quarterly_balance_sheet
    except Exception:
        balance = None
    try:
        cashflow = ticker.quarterly_cashflow
    except Exception:
        cashflow = None

    revenue_now = _row_value(income, "Total Revenue", 0)
    revenue_year_ago = _row_value(income, "Total Revenue", 4)
    net_income_now = _row_value(income, "Net Income", 0)
    net_income_year_ago = _row_value(income, "Net Income", 4)
    operating_income = _row_value(income, "Operating Income", 0)

    total_equity = _row_value(balance, "Total Stockholder Equity", 0) or _row_value(
        balance, "Stockholders Equity", 0
    )
    total_debt = _row_value(balance, "Total Debt", 0)
    total_assets = _row_value(balance, "Total Assets", 0)

    op_cashflow = _row_value(cashflow, "Total Cash From Operating Activities", 0) or _row_value(
        cashflow, "Operating Cash Flow", 0
    )
    capex = _row_value(cashflow, "Capital Expenditures", 0) or _row_value(
        cashflow, "Capital Expenditure", 0
    )
    fcf = None
    if op_cashflow is not None and capex is not None:
        fcf = op_cashflow + capex  # capex is negative in yfinance convention

    market_cap = _safe_float(info.get("marketCap"))
    pe_trailing = _safe_float(info.get("trailingPE"))
    pe_forward = _safe_float(info.get("forwardPE"))
    pb = _safe_float(info.get("priceToBook"))
    roe = _safe_float(info.get("returnOnEquity"))
    if roe is None:
        roe = _ratio(net_income_now, total_equity)
    debt_to_equity_info = _safe_float(info.get("debtToEquity"))
    if debt_to_equity_info is not None:
        # yfinance reports as percentage (e.g. 65 means 0.65)
        debt_to_equity = debt_to_equity_info / 100.0
    else:
        debt_to_equity = _ratio(total_debt, total_equity)

    next_earn = _next_earnings_date(ticker, today)
    days_to_earnings = (next_earn - today).days if next_earn else None

    return {
        "ticker": symbol,
        "as_of_date": today.isoformat(),
        "sector": sector_for(symbol),
        # Income statement
        "revenue_ttm_proxy": revenue_now,
        "revenue_yoy": _yoy(revenue_now, revenue_year_ago),
        "net_income_yoy": _yoy(net_income_now, net_income_year_ago),
        "operating_margin": _ratio(operating_income, revenue_now),
        "net_margin": _ratio(net_income_now, revenue_now),
        # Balance sheet
        "roe": roe,
        "debt_to_equity": debt_to_equity,
        "asset_turnover": _ratio(revenue_now, total_assets),
        # Cash flow
        "fcf_yield": _ratio(fcf, market_cap),
        # Market multiples
        "pe_trailing": pe_trailing,
        "pe_forward": pe_forward,
        "pb": pb,
        "market_cap": market_cap,
        # Earnings event features
        "next_earnings_date": next_earn.isoformat() if next_earn else None,
        "days_to_earnings": days_to_earnings,
        "in_earnings_window_5d": int(days_to_earnings is not None and 0 <= days_to_earnings <= DEFAULT_EVENT_WINDOW_DAYS),
    }


# ----------------------------------------------------------------------------
# Firestore writer
# ----------------------------------------------------------------------------

def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def _valid_firestore_record(record: Dict[str, Any]) -> bool:
    return not _missing(record.get("ticker")) and not _missing(record.get("quarter"))


def _firestore_payload(records: Iterable[Dict[str, Any]] | Dict[str, Any]) -> Dict[str, Any]:
    """Project one company's quarterly records onto the Firestore schema."""
    if isinstance(records, dict):
        records = [records]
    valid = [rec for rec in records if _valid_firestore_record(rec)]
    if not valid:
        raise ValueError("Cannot build fundamentals Firestore payload without ticker and quarter")

    first = valid[0]
    scrape_dates = [str(rec.get("scrape_date")) for rec in valid if not _missing(rec.get("scrape_date"))]
    quarters = {
        str(rec["quarter"]): {
            "quarter_end_date": rec.get("quarter_end_date"),
            "financials": rec.get("financials") or {},
        }
        for rec in valid
    }
    return {
        "company_name": first.get("company_name"),
        "scrape_date": max(scrape_dates) if scrape_dates else None,
        "ticker": first.get("ticker"),
        "quarters": quarters,
    }


def _firestore_doc_id(record: Dict[str, Any]) -> str:
    return str(record["ticker"])


def write_quarterly_to_firestore(
    records: Iterable[Dict[str, Any]],
    collection: str = FIRESTORE_COLLECTION,
    client: Optional[Any] = None,
) -> int:
    """Batch-write company-level fundamentals docs. Returns docs written."""
    client = client or init_firestore_client()

    batch = client.batch()
    pending = 0
    written = 0
    coll = client.collection(collection)
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for rec in records:
        if not _valid_firestore_record(rec):
            continue
        ticker = str(rec["ticker"])
        quarter = str(rec["quarter"])
        grouped.setdefault(ticker, {})[quarter] = rec

    for ticker in sorted(grouped):
        quarter_records = list(grouped[ticker].values())
        doc_ref = coll.document(ticker)
        batch.set(doc_ref, _firestore_payload(quarter_records))
        pending += 1
        written += 1
        if pending >= FIRESTORE_BATCH_SIZE:
            batch.commit()
            batch = client.batch()
            pending = 0

    if pending:
        batch.commit()
    return written


# ----------------------------------------------------------------------------
# Archive helpers — convert nested quarterly records ↔ flat parquet rows
# ----------------------------------------------------------------------------

ARCHIVE_BASE_FIELDS = ("ticker", "company_name", "scrape_date", "quarter", "quarter_end_date")


def _records_to_archive_df(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Encode quarterly records for parquet storage.

    The nested ``financials`` dict is JSON-encoded into a single column so
    pyarrow doesn't have to guess at a struct schema across heterogeneous
    rows (some quarters miss fields).
    """
    rows: List[Dict[str, Any]] = []
    for r in records:
        rows.append(
            {
                "ticker": r.get("ticker"),
                "company_name": r.get("company_name"),
                "scrape_date": r.get("scrape_date"),
                "quarter": r.get("quarter"),
                "quarter_end_date": r.get("quarter_end_date"),
                "financials_json": json.dumps(r.get("financials") or {}, default=str),
            }
        )
    return pd.DataFrame(rows)


def _archive_df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Reverse of :func:`_records_to_archive_df` — re-nest the financials."""
    out: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        raw_fin = row.get("financials_json") or "{}"
        try:
            financials = json.loads(raw_fin) if isinstance(raw_fin, str) else {}
        except (json.JSONDecodeError, TypeError):
            financials = {}
        out.append(
            {
                "ticker": row.get("ticker"),
                "company_name": row.get("company_name"),
                "scrape_date": row.get("scrape_date"),
                "quarter": row.get("quarter"),
                "quarter_end_date": row.get("quarter_end_date"),
                "financials": financials,
            }
        )
    return out


def _upload_archive_df(df: pd.DataFrame, client: Optional[Any] = None) -> int:
    """Adapter that lets the shared archive helpers upload via our records writer."""
    records = _archive_df_to_records(df)
    invalid = [rec for rec in records if not _valid_firestore_record(rec)]
    if invalid:
        raise ValueError(f"{len(invalid)} fundamentals archive rows are missing ticker or quarter")
    expected_docs = len({str(rec["ticker"]) for rec in records})
    written_docs = write_quarterly_to_firestore(records, client=client)
    if written_docs != expected_docs:
        raise RuntimeError(f"Firestore wrote {written_docs}/{expected_docs} fundamentals company docs")
    return len(df)


# ----------------------------------------------------------------------------
# Refresh orchestration
# ----------------------------------------------------------------------------

def refresh_fundamentals(
    tickers: Iterable[str],
    output_path: Path = DEFAULT_OUTPUT,
    today: Optional[date] = None,
    sleep_seconds: float = 0.3,
    write_firestore: bool = True,
    firestore_client: Optional[Any] = None,
    lookback_quarters: int = DEFAULT_LOOKBACK_QUARTERS,
    archive_path: Optional[Path] = None,
    flat_snapshot_path: Optional[Path] = None,
    collect: bool = True,
) -> Dict[str, Any]:
    """Refresh fundamentals with the fall-through retry-queue pattern.

    Flow (see :mod:`ingestion._archive` for the contract):

    1. **Drain** — try to upload any retained archive parquet from a prior
       failed run. Successful drains delete the file; failures retain it.
    2. **Collect** — when enabled, pulls quarterly records via yfinance,
       regardless of drain outcome.
    3. **Merge-stage** — overlays today's records on top of any retained
       archive rows, deduping on ``(ticker, quarter)``.
    4. **Upload + clear** — pushes the staged archive to Firestore; deletes
       on success, retains on failure for the next run.

    Returns ``{flat_rows, quarterly_records, firestore_writes, archive_drained}``.
    """
    if archive_path is None:
        archive_path = archive_path_for(output_path)
    upload_fn = lambda df: _upload_archive_df(df, client=firestore_client)

    # ---- Step 1: drain pending archive (runs even if collection is off) -
    archive_drained = 0
    if write_firestore:
        drained = drain_pending_archive(archive_path, upload_fn, _log)
        if drained is not None:
            archive_drained = int(drained)
    elif archive_path.exists():
        _log("[fundamentals] pending archive present but --no-firestore set; leaving file alone")

    # ---- Skip-collect short-circuit (COLLECT_FUNDAMENTALS=false) --------
    if not collect:
        _log("[fundamentals] COLLECT_FUNDAMENTALS=false; skipping fresh collection (drain-only run)")
        existing_quarterly = load_fundamentals(archive_path)
        return {
            "flat_rows": 0,
            "quarterly_records": len(existing_quarterly),
            "firestore_writes": 0,
            "archive_drained": archive_drained,
        }

    # ---- Step 2: fresh collection ---------------------------------------
    today = today or date.today()
    flat_rows: List[Dict[str, Any]] = []
    quarterly_records: List[Dict[str, Any]] = []
    for symbol in tickers:
        try:
            flat_rows.append(fetch_fundamentals(symbol, today=today))
        except Exception as exc:
            _log(f"[fundamentals] flat-snapshot {symbol} failed: {exc}")

        try:
            qr = fetch_quarterly_records(symbol, today=today, lookback=lookback_quarters)
            quarterly_records.extend(qr)
            _log(f"[fundamentals] {symbol}: {len(qr)} quarterly records")
        except Exception as exc:
            _log(f"[fundamentals] quarterly {symbol} failed: {exc}")

        time_module.sleep(sleep_seconds)

    # ---- Step 3: prepare quarterly records for Firestore staging
    quarterly_df = _records_to_archive_df(quarterly_records) if quarterly_records else pd.DataFrame()
    if not quarterly_df.empty:
        _log(f"[fundamentals] prepared {len(quarterly_df)} quarterly rows for Firestore staging")
    else:
        _log("[fundamentals] no quarterly records fetched")

    # Keep the flat snapshot in memory for summary only; it is no longer
    # persisted because Firestore-bound ingest outputs are staged under archive.
    flat_df = pd.DataFrame(flat_rows)
    if not flat_df.empty:
        flat_df["pe_sector_pct"] = (
            flat_df.groupby("sector")["pe_trailing"].transform(lambda s: s.rank(pct=True))
        )

    # ---- Steps 4 + 5: merge-stage and upload ----------------------------
    firestore_writes = 0
    merged = merge_stage_to_archive(
        quarterly_df, archive_path, dedup_keys=("ticker", "quarter"), log=_log
    )
    if write_firestore:
        if merged > 0:
            firestore_writes = upload_and_clear_archive(archive_path, upload_fn, _log)
    else:
        _log("[fundamentals] Firestore write skipped (--no-firestore); staged archive retained")

    return {
        "flat_rows": len(flat_df),
        "quarterly_records": len(quarterly_records),
        "firestore_writes": firestore_writes,
        "archive_drained": archive_drained,
    }


def load_fundamentals(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def cache_is_fresh(path: Path = DEFAULT_OUTPUT, max_age_days: int = 7) -> bool:
    if not path.exists():
        return False
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return age <= timedelta(days=max_age_days)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh quarterly fundamentals from yfinance.")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers (defaults to NIFTY 50).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output parquet path.")
    parser.add_argument("--sleep", type=float, default=0.3, help="Sleep between tickers (seconds).")
    parser.add_argument(
        "--lookback-quarters",
        type=int,
        default=DEFAULT_LOOKBACK_QUARTERS,
        help="How many recent quarters to write to Firestore per ticker.",
    )
    parser.add_argument(
        "--no-firestore",
        action="store_true",
        help="Skip Firestore writes (parquet snapshot only).",
    )
    args = parser.parse_args(argv)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = list_tickers()

    summary = refresh_fundamentals(
        tickers,
        output_path=Path(args.output),
        sleep_seconds=args.sleep,
        write_firestore=not args.no_firestore,
        lookback_quarters=args.lookback_quarters,
        collect=truthy_env("COLLECT_FUNDAMENTALS", default=True),
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
