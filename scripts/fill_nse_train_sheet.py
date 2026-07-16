#!/usr/bin/env python3
"""Fill the legacy-format NSE training sheet with OHLCV, then train the model.

This sheet (``nse_stock_data_train``) stores OHLCV in **legacy per-ticker
columns** — ``Open_<TICKER>.NS``, ``High_<TICKER>.NS``, ``Low_<TICKER>.NS``,
``Close_<TICKER>.NS``, ``Volume_<TICKER>.NS`` — with ``Date_`` / ``Date_str``
date columns. (The standard ``Open/High/Low/Close/Adj Close/Volume`` columns
exist in the header but are blank in every row and are NOT what the pipeline
reads.) The stored close is **adjusted** (yfinance ``auto_adjust=True``).

Step 1 — Fill: for each ticker worksheet, download adjusted daily OHLCV over
``[--start, --end]`` (default 2014-01-01 → 2026-06-30) and append rows for any
``Date_str`` not already present (full-range fetch ⇒ back-fills 2014 and
forward-fills to the end). Only the date + legacy OHLCV cells are written; the
indicator / forecast columns are left blank — the trainer recomputes indicators
from OHLCV, so the model still learns fully from the new rows.

Step 2 — Train: fine-tune the production ensemble on the sheet's full history
(``monthly_finetune`` with ``--no-fine-tune-batch-only-new-data`` ⇒ all rows),
warm-starting from and overwriting ``outputs/Saved_Models``.

Usage:
    # print-only self-test on one ticker (no writes):
    python scripts/fill_nse_train_sheet.py --tickers TCS --dry-run --no-train
    # real fill (all tickers) + train:
    python scripts/fill_nse_train_sheet.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ingestion.aliases import list_tickers  # noqa: E402

DEFAULT_SHEET_ID = "1_gpRclY80tH3r54b9R5HTXqKF7R7bnMoWugF9Fy-boI"
DEFAULT_START = "2014-01-01"
DEFAULT_END = "2026-06-30"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
APPEND_BATCH = 500


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Sheet / schema
# ---------------------------------------------------------------------------

def _client(credentials_path: str):
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    return gspread.authorize(creds)


def _legacy_indices(header: List[str], title: str) -> Optional[Dict[str, int]]:
    """Locate the legacy date + OHLCV column indices for a worksheet.

    Returns a mapping ``{Date_, Date_str, Open, High, Low, Close, Volume} -> idx``
    or ``None`` if the legacy OHLCV columns aren't present.
    """
    idx = {name: i for i, name in enumerate(header)}
    out: Dict[str, int] = {}
    # date columns
    if "Date_" in idx:
        out["Date_"] = idx["Date_"]
    elif "Date" in idx:
        out["Date_"] = idx["Date"]
    if "Date_str" in idx:
        out["Date_str"] = idx["Date_str"]
    # legacy OHLCV: prefer exact "<Field>_<TITLE>.NS", else any "<Field>_*.NS"
    for field in ("Open", "High", "Low", "Close", "Volume"):
        exact = f"{field}_{title}.NS"
        if exact in idx:
            out[field] = idx[exact]
            continue
        pat = re.compile(rf"^{field}_.+\.NS$", re.IGNORECASE)
        match = next((i for i, h in enumerate(header) if pat.match(str(h))), None)
        if match is not None:
            out[field] = match
    required = {"Date_str", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(out.keys()):
        return None
    return out


def _download_adjusted(symbol: str, start: date, end: date, interval: str):
    """Adjusted daily OHLCV (auto_adjust=True) as a tidy frame with Date_str."""
    import pandas as pd
    import yfinance as yf

    raw = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        auto_adjust=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    # yfinance may return a column MultiIndex for a single symbol → flatten.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    df = raw.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["Date_str"] = df[date_col].dt.strftime("%Y-%m-%d")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset=["Date_str"], keep="last").sort_values("Date_str")
    return df[["Date_str", "Open", "High", "Low", "Close", "Volume"]]


def _build_rows(df_new, header_len: int, cols: Dict[str, int]) -> List[List[Any]]:
    """Full-width rows with only date + legacy OHLCV cells filled."""
    rows: List[List[Any]] = []
    for _, r in df_new.iterrows():
        row: List[Any] = [""] * header_len
        ds = str(r["Date_str"])
        if "Date_" in cols:
            row[cols["Date_"]] = f"{ds} 00:00:00"
        row[cols["Date_str"]] = ds
        row[cols["Open"]] = float(r["Open"])
        row[cols["High"]] = float(r["High"])
        row[cols["Low"]] = float(r["Low"])
        row[cols["Close"]] = float(r["Close"])
        row[cols["Volume"]] = int(float(r["Volume"]))
        rows.append(row)
    return rows


def fill_worksheet(ws, title: str, start: date, end: date, interval: str, dry_run: bool) -> Dict[str, Any]:
    symbol = f"{title}.NS"
    values = ws.get_all_values()
    if not values:
        return {"ticker": title, "appended": 0, "status": "empty_worksheet_no_header"}
    header = [str(h) for h in values[0]]
    cols = _legacy_indices(header, title)
    if cols is None:
        return {"ticker": title, "appended": 0, "status": "no_legacy_ohlcv_columns"}

    ds_idx = cols["Date_str"]
    existing = {str(row[ds_idx]).strip() for row in values[1:] if ds_idx < len(row) and str(row[ds_idx]).strip()}

    df = _download_adjusted(symbol, start, end, interval)
    if df.empty:
        return {"ticker": title, "appended": 0, "status": "no_data_from_yfinance"}

    df_new = df[~df["Date_str"].isin(existing)].copy()
    if df_new.empty:
        return {"ticker": title, "appended": 0, "skipped_existing": len(existing), "status": "already_present"}

    rows = _build_rows(df_new, len(header), cols)
    if dry_run:
        sample = rows[0] if rows else []
        filled = {header[i]: sample[i] for i in sorted(cols.values())} if sample else {}
        return {
            "ticker": title, "appended": 0, "would_append": len(rows),
            "skipped_existing": len(existing),
            "new_range": [df_new["Date_str"].min(), df_new["Date_str"].max()],
            "first_row_filled_cells": filled, "status": "dry_run",
        }

    for i in range(0, len(rows), APPEND_BATCH):
        batch = rows[i:i + APPEND_BATCH]
        ws.append_rows(batch, value_input_option="RAW")
    return {"ticker": title, "appended": len(rows), "skipped_existing": len(existing), "status": "filled"}


def fill_sheet(sheet_id, tickers, start, end, interval, credentials_path, sleep_seconds, dry_run) -> Dict[str, Any]:
    gc = _client(credentials_path)
    ss = gc.open_by_key(sheet_id)
    by_title = {w.title.strip().upper(): w for w in ss.worksheets()}
    results, total = [], 0
    for i, t in enumerate(tickers, 1):
        ws = by_title.get(t.upper())
        if ws is None:
            results.append({"ticker": t, "appended": 0, "status": "worksheet_missing"})
            log(f"[{i}/{len(tickers)}] {t}: worksheet_missing")
            continue
        res = fill_worksheet(ws, t.upper(), start, end, interval, dry_run)
        results.append(res)
        total += res.get("appended", 0)
        log(f"[{i}/{len(tickers)}] {t}: {res['status']} "
            f"(appended={res.get('appended', 0)}, would_append={res.get('would_append', 0)})")
        if sleep_seconds and not dry_run:
            time.sleep(sleep_seconds)
    return {"step": "fill", "sheet_id": sheet_id, "tickers": len(tickers),
            "total_rows_appended": total, "per_ticker": results}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_production_model(sheet_id, credentials_path, device, state_file, check_only=False) -> Dict[str, Any]:
    import monthly_finetune as mf

    argv = [
        "--operational-sheet-id", sheet_id,
        "--historical-sheet-id", sheet_id,
        "--model-dir", str(_REPO_ROOT / "outputs" / "Saved_Models"),
        "--metadata", str(_REPO_ROOT / "outputs" / "pipeline_metadata.json"),
        "--state-file", state_file,
        "--output-dir", str(_REPO_ROOT / "outputs" / "monthly_finetune"),
        "--no-fine-tune-batch-only-new-data",
        "--skip-archival",
        "--device", device,
    ]
    if credentials_path:
        argv += ["--google-credentials", credentials_path]
    if check_only:
        argv += ["--check-only"]

    saved = sys.argv
    try:
        sys.argv = ["monthly_finetune.py", *argv]
        args = mf.parse_args()
    finally:
        sys.argv = saved
    result = mf.run_monthly_finetune(args)
    result["step"] = "train"
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pd(v: str) -> date:
    return datetime.strptime(v, "%Y-%m-%d").date()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sheet-id", default=DEFAULT_SHEET_ID)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--tickers", default=None, help="Comma-separated subset (default: full NIFTY universe).")
    p.add_argument("--interval", default="1d")
    p.add_argument("--google-credentials", required=True, help="Service-account JSON path.")
    p.add_argument("--sleep", type=float, default=0.4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--state-file", default=str(_REPO_ROOT / "state" / "full_train_state.json"))
    p.add_argument("--dry-run", action="store_true", help="Fill step prints what it would append; no writes.")
    p.add_argument("--check-only", action="store_true", help="Train step evaluates the gate/dataset only; no training.")
    p.add_argument("--no-fill", action="store_true")
    p.add_argument("--no-train", action="store_true")
    args = p.parse_args(argv)

    start, end = _pd(args.start), _pd(args.end)
    if start > end:
        p.error("--start must be <= --end")
    tickers = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
               if args.tickers else list_tickers())

    summary: Dict[str, Any] = {"sheet_id": args.sheet_id, "range": [args.start, args.end], "tickers": len(tickers)}

    if not args.no_fill:
        log(f"=== FILL{' (dry-run)' if args.dry_run else ''}: {len(tickers)} tickers, {args.start} → {args.end} ===")
        summary["fill"] = fill_sheet(args.sheet_id, tickers, start, end, args.interval,
                                     args.google_credentials, args.sleep, args.dry_run)
        log(f"=== FILL done: {summary['fill']['total_rows_appended']} rows appended ===")
    else:
        log("=== FILL skipped ===")

    if not args.no_train:
        log(f"=== TRAIN{' (check-only)' if args.check_only else ''}: production model on full sheet history ===")
        summary["train"] = train_production_model(args.sheet_id, args.google_credentials,
                                                   args.device, args.state_file, args.check_only)
        log(f"=== TRAIN done: status={summary['train'].get('status')} ===")
    else:
        log("=== TRAIN skipped ===")

    print(json.dumps(summary, indent=2, default=str))
    return 1 if (summary.get("train") or {}).get("status") == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
