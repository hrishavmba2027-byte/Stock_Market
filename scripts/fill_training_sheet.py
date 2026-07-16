#!/usr/bin/env python3
"""Standalone: fill a Google Sheet with OHLCV history, then train + save the model.

What it does (two independent steps, either can be skipped):

1. **Fill** — for every NIFTY-universe ticker, download daily OHLCV from yfinance
   over ``[--start, --end]`` (default 2014-01-01 → 2026-06-30) and write one
   worksheet per ticker to the target sheet using the standard column schema
   (``Date, Date_str, Open, High, Low, Close, Adj Close, Volume``). Worksheets
   that don't exist yet are created. **Already-present dates are skipped** — only
   missing rows are appended (full-range fetch, so gaps are back-filled too).

2. **Train** — fine-tune the production ensemble on this sheet's full history
   (``--no-fine-tune-batch-only-new-data`` ⇒ all rows, not just recent) by
   driving ``monthly_finetune.py``. It warm-starts from the existing production
   checkpoints in ``outputs/Saved_Models`` and atomically overwrites them there
   (plus ``outputs/pipeline_metadata.json``). Technical indicators are computed
   in-code at train time (``Feature_Engineering.compute_indicators``), so the
   sheet only needs OHLCV.

Reuses the repo's own helpers (``Data_update`` for fetch/append,
``monthly_finetune`` for training) so the data + model stay format-compatible.

Examples
--------
    # Fill + train (full universe, full range) — writes to Firestore-free sheet:
    python scripts/fill_training_sheet.py

    # Only fill, a couple of tickers, custom range:
    python scripts/fill_training_sheet.py --no-train --tickers TCS,RELIANCE \
        --start 2014-01-01 --end 2026-06-30

    # Only train from an already-filled sheet:
    python scripts/fill_training_sheet.py --no-fill
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Make sure the repo root is importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import Data_update as du  # noqa: E402
from ingestion.aliases import list_tickers  # noqa: E402

# The sheet the user provided.
DEFAULT_SHEET_ID = "1_gpRclY80tH3r54b9R5HTXqKF7R7bnMoWugF9Fy-boI"
DEFAULT_START = "2014-01-01"
DEFAULT_END = "2026-06-30"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

def _open_spreadsheet(sheet_id: str, credentials_path: Optional[str]) -> Any:
    client = du.authorize_gspread(credentials_path)
    return du.with_retry(lambda: client.open_by_key(sheet_id), f"open spreadsheet {sheet_id}")


def _ensure_worksheet(spreadsheet: Any, title: str, needed_rows: int) -> Tuple[Any, bool]:
    """Return ``(worksheet, is_new)``, creating + header-initializing if absent."""
    for ws in du.with_retry(spreadsheet.worksheets, "list worksheets"):
        if str(ws.title).strip().upper() == title.upper():
            return ws, False
    cols = len(du.REQUIRED_COLUMNS) + 2
    rows = max(needed_rows + 50, 100)
    ws = du.with_retry(
        lambda: spreadsheet.add_worksheet(title=title, rows=rows, cols=cols),
        f"add worksheet {title}",
    )
    du.initialize_empty_sheet(ws)  # writes the REQUIRED_COLUMNS header row
    log(f"[{title}] created worksheet")
    return ws, True


def _existing_state(ws: Any, title: str, is_new: bool):
    """Return ``(existing_date_strs, append_headers, append_schema)``."""
    if is_new:
        return set(), None, None
    frame, headers, is_empty = du.worksheet_to_frame(ws)
    if is_empty:
        return set(), None, None
    schema, schema_error = du.resolve_sheet_schema(headers, title.upper())
    if schema is None:
        # Header we don't recognise → append in REQUIRED_COLUMNS order.
        log(f"[{title}] unrecognised header ({schema_error}); appending in default column order")
        return set(), None, None
    normalized = du.normalize_frame_to_required_columns(frame, headers, schema)
    if normalized.empty:
        existing: Set[str] = set()
    else:
        _, existing = du.normalize_existing_dates(normalized)
    return existing, headers, schema


def fill_worksheet(
    spreadsheet: Any,
    ticker: str,
    start: date,
    end: date,
    interval: str,
) -> Dict[str, Any]:
    stock = ticker.strip().upper()
    symbol = f"{stock}.NS"
    end_exclusive = end + timedelta(days=1)
    try:
        raw = du.with_retry(
            lambda: du.download_stock_data(symbol, start, end_exclusive, interval),
            f"{symbol}: yfinance download",
            attempts=3,
            backoff=1.5,
        )
    except Exception as exc:  # noqa: BLE001 — per-ticker isolation
        return {"ticker": stock, "appended": 0, "status": f"download_failed: {exc}"}

    if raw is None or raw.empty:
        return {"ticker": stock, "appended": 0, "status": "no_data_from_yfinance"}

    try:
        df_all = du.format_yfinance_frame(raw)
    except Exception as exc:  # noqa: BLE001
        return {"ticker": stock, "appended": 0, "status": f"malformed_yfinance: {exc}"}

    ws, is_new = _ensure_worksheet(spreadsheet, stock, needed_rows=len(df_all))
    existing, append_headers, append_schema = _existing_state(ws, stock, is_new)

    df_new = df_all[~df_all["Date_str"].isin(existing)].copy()
    df_new = df_new.drop_duplicates(subset=["Date_str"], keep="last").sort_values("Date_str")
    if df_new.empty:
        return {"ticker": stock, "appended": 0, "status": "already_present"}

    appended = du.append_rows(ws, df_new, headers=append_headers, schema=append_schema)
    return {
        "ticker": stock,
        "appended": int(appended),
        "skipped_existing": int(len(existing)),
        "status": "filled",
    }


def fill_sheet(
    sheet_id: str,
    tickers: List[str],
    start: date,
    end: date,
    interval: str,
    credentials_path: Optional[str],
    sleep_seconds: float,
) -> Dict[str, Any]:
    spreadsheet = _open_spreadsheet(sheet_id, credentials_path)
    results: List[Dict[str, Any]] = []
    total_appended = 0
    for i, ticker in enumerate(tickers, 1):
        res = fill_worksheet(spreadsheet, ticker, start, end, interval)
        results.append(res)
        total_appended += res["appended"]
        log(f"[{i}/{len(tickers)}] {res['ticker']}: {res['status']} "
            f"(appended={res['appended']})")
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return {
        "step": "fill",
        "sheet_id": sheet_id,
        "tickers": len(tickers),
        "total_rows_appended": total_appended,
        "per_ticker": results,
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_production_model(
    sheet_id: str,
    credentials_path: Optional[str],
    device: str,
    state_file: str,
) -> Dict[str, Any]:
    """Fine-tune + save the PRODUCTION ensemble on the full sheet history.

    Drives monthly_finetune with the target sheet as both the operational and
    historical source, ``--no-fine-tune-batch-only-new-data`` (train on ALL
    rows), and ``--skip-archival`` (no row shuffling). Warm-starts from and
    overwrites ``outputs/Saved_Models``.
    """
    import monthly_finetune as mf

    argv = [
        "--operational-sheet-id", sheet_id,
        "--historical-sheet-id", sheet_id,
        "--model-dir", str(_REPO_ROOT / "outputs" / "Saved_Models"),
        "--metadata", str(_REPO_ROOT / "outputs" / "pipeline_metadata.json"),
        "--state-file", state_file,
        "--output-dir", str(_REPO_ROOT / "outputs" / "monthly_finetune"),
        "--no-fine-tune-batch-only-new-data",  # train on the full history
        "--skip-archival",
        "--device", device,
    ]
    if credentials_path:
        argv += ["--google-credentials", credentials_path]

    # monthly_finetune.parse_args() reads sys.argv directly (no argv param).
    saved_argv = sys.argv
    try:
        sys.argv = ["monthly_finetune.py", *argv]
        args = mf.parse_args()
    finally:
        sys.argv = saved_argv
    result = mf.run_monthly_finetune(args)
    result["step"] = "train"
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet-id", default=DEFAULT_SHEET_ID, help="Target Google Sheet id.")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD (default 2014-01-01).")
    parser.add_argument("--end", default=DEFAULT_END, help="End date YYYY-MM-DD (default 2026-06-30).")
    parser.add_argument("--tickers", default=None, help="Comma-separated subset (default: full NIFTY universe).")
    parser.add_argument("--interval", default="1d", help="yfinance interval (default 1d).")
    parser.add_argument("--google-credentials", default=None, help="Service-account JSON path (else env / default).")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between tickers (rate-limit politeness).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--state-file",
        default=str(_REPO_ROOT / "state" / "full_train_state.json"),
        help="Fine-tune state file (a fresh one keeps this run independent of the monthly cadence).",
    )
    parser.add_argument("--no-fill", action="store_true", help="Skip the sheet-fill step.")
    parser.add_argument("--no-train", action="store_true", help="Skip the training step.")
    args = parser.parse_args(argv)

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if start > end:
        parser.error("--start must be <= --end")

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else list_tickers()
    )

    summary: Dict[str, Any] = {"sheet_id": args.sheet_id, "range": [args.start, args.end], "tickers": len(tickers)}

    if not args.no_fill:
        log(f"=== FILL: {len(tickers)} tickers, {args.start} → {args.end} into {args.sheet_id} ===")
        summary["fill"] = fill_sheet(
            args.sheet_id, tickers, start, end, args.interval,
            args.google_credentials, args.sleep,
        )
        log(f"=== FILL done: {summary['fill']['total_rows_appended']} rows appended ===")
    else:
        log("=== FILL skipped (--no-fill) ===")

    if not args.no_train:
        log("=== TRAIN: fine-tuning production model on full sheet history ===")
        summary["train"] = train_production_model(
            args.sheet_id, args.google_credentials, args.device, args.state_file,
        )
        log(f"=== TRAIN done: status={summary['train'].get('status')} ===")
    else:
        log("=== TRAIN skipped (--no-train) ===")

    print(json.dumps(summary, indent=2, default=str))
    train_status = (summary.get("train") or {}).get("status", "ok")
    return 1 if train_status == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
