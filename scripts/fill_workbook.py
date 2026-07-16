#!/usr/bin/env python3
"""Complete the local NSE training workbook: fill 2014-2026, no empty feature cells.

Reads ``Data/archive/nse_stock_data_train.xlsx`` (49 legacy-schema sheets, one
per ticker) and writes a completed copy where, for every ticker:

* OHLCV covers the full ``[--start, --end]`` range (default 2014-01-01 →
  2026-06-30). Existing dated rows are **kept as-is**; only missing dates are
  added (adjusted yfinance download), so nothing already present is overwritten.
* Every **feature** column is (re)computed on the full price series and filled —
  the 28 technical indicators (``Feature_Engineering.INDICATOR_COLUMNS``), the 11
  decision features (``DECISION_FEATURE_COLUMNS``), plus the standard OHLCV mirror
  columns — so there are no empty feature cells (leading indicator-warmup NaNs are
  back-filled). ``Row_finetuned`` is set to 1.

The forecast/prediction columns (``Forecast_Close_T+*``, ``predicted``,
``Predicted_Close_Price``) are model OUTPUTS and are left for the post-training
inference pass — they cannot be filled before the model is trained.

Output goes to ``--out`` (default ``..._filled.xlsx``) to preserve the original.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Feature_Engineering import (  # noqa: E402
    DECISION_FEATURE_COLUMNS,
    INDICATOR_COLUMNS,
    compute_decision_features,
    compute_indicators,
)

DEFAULT_IN = _REPO_ROOT / "Data" / "archive" / "nse_stock_data_train.xlsx"
DEFAULT_OUT = _REPO_ROOT / "Data" / "archive" / "nse_stock_data_train_filled.xlsx"
DEFAULT_START = "2014-01-01"
DEFAULT_END = "2026-06-30"

STD_OHLCV = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
FORECAST_COLS = [f"Forecast_Close_T+{h}" for h in range(1, 16)]
OUTPUT_COLS = ["predicted", "Predicted_Close_Price", *FORECAST_COLS]


def log(msg: str) -> None:
    print(msg, flush=True)


def _legacy_cols(header: List[str], title: str) -> Optional[Dict[str, str]]:
    """Map canonical OHLCV -> the legacy column name for this ticker."""
    hset = set(header)
    out: Dict[str, str] = {}
    for field in ("Open", "High", "Low", "Close", "Volume"):
        name = f"{field}_{title}.NS"
        if name in hset:
            out[field] = name
    if len(out) == 5:
        return out
    return None


def _download_adjusted(symbol: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                      end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                      interval="1d", progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    df = raw.reset_index()
    dcol = "Date" if "Date" in df.columns else df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol])
    df["Date_str"] = df[dcol].dt.strftime("%Y-%m-%d")
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c not in df.columns:
            return pd.DataFrame()
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df[["Date_str", "Open", "High", "Low", "Close", "Volume"]].drop_duplicates("Date_str")


def _existing_ohlcv(df: pd.DataFrame, legacy: Dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Date_str"] = df["Date_str"].astype(str)
    for field, col in legacy.items():
        out[field] = pd.to_numeric(df[col], errors="coerce")
    return out.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).drop_duplicates("Date_str")


def build_completed_sheet(df: pd.DataFrame, title: str, start: date, end: date) -> Dict[str, object]:
    header = list(df.columns)
    legacy = _legacy_cols(header, title)
    if legacy is None:
        return {"status": "no_legacy_cols", "df": df}

    existing = _existing_ohlcv(df, legacy)
    downloaded = _download_adjusted(f"{title}.NS", start, end)
    if downloaded.empty and existing.empty:
        return {"status": "no_ohlcv", "df": df}

    # Union of dates: keep existing OHLCV; add downloaded rows for missing dates.
    have = set(existing["Date_str"])
    new_rows = downloaded[~downloaded["Date_str"].isin(have)] if not downloaded.empty else pd.DataFrame()
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.sort_values("Date_str").reset_index(drop=True)
    n_added = int(len(new_rows))

    # ── recompute features on the full, continuous series ────────────────────
    feat_in = combined.rename(columns={}).copy()  # has Open/High/Low/Close/Volume + Date_str
    engineered = compute_indicators(feat_in.copy())
    decision = compute_decision_features(feat_in.copy())

    out = pd.DataFrame()
    out["Date_str"] = combined["Date_str"]
    out["Date_"] = combined["Date_str"] + " 00:00:00"
    for field, col in legacy.items():
        out[col] = combined[field]
    # standard OHLCV mirror (Adj Close == adjusted Close)
    out["Open"] = combined["Open"]; out["High"] = combined["High"]
    out["Low"] = combined["Low"]; out["Close"] = combined["Close"]
    out["Adj Close"] = combined["Close"]; out["Volume"] = combined["Volume"]
    # indicators (align by position — same row order as combined)
    for col in INDICATOR_COLUMNS:
        if col in engineered.columns:
            out[col] = engineered[col].values
    for col in DECISION_FEATURE_COLUMNS:
        if col in decision.columns:
            out[col] = decision[col].values
    out["Row_finetuned"] = 1

    # Carry over any forecast/output columns from the source (matched by Date_str);
    # they stay empty for now (filled post-training by the inference pass).
    src = df.set_index("Date_str")
    for col in OUTPUT_COLS:
        if col in src.columns:
            out[col] = out["Date_str"].map(src[col]).values

    # Back-fill leading indicator-warmup NaNs so there are no empty feature cells.
    num_feats = [c for c in INDICATOR_COLUMNS if c in out.columns]
    out[num_feats] = out[num_feats].bfill().ffill()
    dec_feats = [c for c in DECISION_FEATURE_COLUMNS if c in out.columns]
    out[dec_feats] = out[dec_feats].replace("", np.nan).bfill().ffill().fillna("")

    # Reindex to the original column order; any original column not built stays.
    for col in header:
        if col not in out.columns:
            out[col] = df.set_index("Date_str").reindex(out["Date_str"])[col].values if col in df.columns else ""
    out = out[header]
    return {"status": "ok", "df": out, "rows": len(out), "added": n_added,
            "range": [out["Date_str"].min(), out["Date_str"].max()]}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--tickers", default=None, help="Comma-separated subset (default: all sheets).")
    args = p.parse_args(argv)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    xl = pd.ExcelFile(args.inp)
    sheets = xl.sheet_names
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",")}
        sheets = [s for s in sheets if s.upper() in wanted]

    log(f"=== FILL WORKBOOK: {len(sheets)} sheets, {args.start} → {args.end} ===")
    results = {}
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        for i, sheet in enumerate(sheets, 1):
            df = pd.read_excel(xl, sheet_name=sheet)
            if "Date_str" not in df.columns:
                df.to_excel(writer, sheet_name=sheet, index=False)
                log(f"[{i}/{len(sheets)}] {sheet}: no Date_str, copied as-is")
                continue
            res = build_completed_sheet(df, sheet.upper(), start, end)
            out_df = res["df"]
            out_df.to_excel(writer, sheet_name=sheet, index=False)
            results[sheet] = {k: v for k, v in res.items() if k != "df"}
            log(f"[{i}/{len(sheets)}] {sheet}: {res['status']} "
                f"rows={res.get('rows','?')} added={res.get('added','?')} range={res.get('range','?')}")
    log(f"=== DONE → {args.out} ===")
    import json
    print(json.dumps({"sheets": len(sheets), "out": args.out, "per_sheet": results}, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
