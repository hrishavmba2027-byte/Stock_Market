"""P1.6 — Cross-sectional and regime features.

Single-ticker indicators in ``Feature_Engineering.py`` only see a stock in
isolation. At the 1–4 week horizon, relative-strength vs the index, regime
(VIX, NIFTY trend), and cross-sectional rank explain more of the dispersion
than another momentum indicator on the same ticker would.

This module operates on a dict ``{ticker: DataFrame}`` already produced by
``Feature_Engineering.compute_indicators``. It returns the same dict with
new columns added.

Index data is pulled live via ``yfinance`` (``^NSEI`` for NIFTY 50,
``^INDIAVIX`` for INDIAVIX) and cached to ``Data/archive/indices.parquet``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_INDEX_CACHE = Path(__file__).resolve().parents[1] / "Data" / "archive" / "indices.parquet"
NIFTY_SYMBOL = "^NSEI"
VIX_SYMBOL = "^INDIAVIX"
FIRESTORE_COLLECTION = "index_history"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def fetch_index_history(
    start: str = "2015-01-01",
    end: Optional[str] = None,
    output_path: Path = DEFAULT_INDEX_CACHE,
) -> pd.DataFrame:
    import yfinance as yf

    end = end or (date.today() + timedelta(days=1)).isoformat()
    frames: List[pd.DataFrame] = []
    for symbol, label in [(NIFTY_SYMBOL, "nifty"), (VIX_SYMBOL, "vix")]:
        try:
            df = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
            if df is None or df.empty:
                _log(f"[cross-sectional] {symbol} returned empty frame")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df[["Close"]].rename(columns={"Close": f"{label}_close"})
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            frames.append(df)
        except Exception as exc:
            _log(f"[cross-sectional] {symbol} fetch failed: {exc}")

    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="outer")
    merged = merged.sort_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path)
    _log(f"[cross-sectional] wrote {len(merged)} index rows to {output_path}")
    return merged


def load_index_history(path: Path = DEFAULT_INDEX_CACHE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.sort_index()
    return df


def write_indices_to_firestore(
    df: pd.DataFrame,
    collection: str = FIRESTORE_COLLECTION,
    client: Optional[Any] = None,
) -> int:
    """Wipe ``collection`` and write one doc per date. Returns rows written."""
    if df.empty:
        return 0
    from ingestion._firestore import batch_write, init_firestore_client, wipe_collection

    client = client or init_firestore_client()
    wipe_collection(client, collection)

    def _docs():
        for date_val, row in df.iterrows():
            doc_id = pd.Timestamp(date_val).strftime("%Y-%m-%d")
            payload: Dict[str, Any] = {"date": doc_id}
            for col in df.columns:
                val = row[col]
                payload[col] = None if pd.isna(val) else float(val)
            yield doc_id, payload

    written = batch_write(client, collection, _docs())
    _log(f"[cross-sectional] wrote {written} rows to Firestore collection '{collection}'")
    return written


def _compute_regime(index_df: pd.DataFrame) -> pd.DataFrame:
    if index_df.empty:
        return pd.DataFrame()
    df = index_df.copy()
    df["nifty_log_return"] = np.log(df["nifty_close"]).diff()
    df["nifty_return_20d"] = df["nifty_close"].pct_change(20)
    df["nifty_vol_20d"] = df["nifty_log_return"].rolling(20).std() * np.sqrt(252)
    df["nifty_trend_50d_200d"] = (
        df["nifty_close"].rolling(50).mean() - df["nifty_close"].rolling(200).mean()
    ) / df["nifty_close"].rolling(200).mean()
    if "vix_close" in df.columns:
        df["vix_level"] = df["vix_close"]
        df["vix_delta_5d"] = df["vix_close"].diff(5)
    else:
        df["vix_level"] = np.nan
        df["vix_delta_5d"] = np.nan
    df["regime_high_vol"] = (df["vix_level"] > df["vix_level"].rolling(252).median()).astype("Int64")
    df["regime_nifty_up"] = (df["nifty_trend_50d_200d"] > 0).astype("Int64")
    keep = [
        "nifty_close",
        "nifty_log_return",
        "nifty_return_20d",
        "nifty_vol_20d",
        "nifty_trend_50d_200d",
        "vix_level",
        "vix_delta_5d",
        "regime_high_vol",
        "regime_nifty_up",
    ]
    return df[[c for c in keep if c in df.columns]]


def _attach_per_stock(df: pd.DataFrame, regime: pd.DataFrame, date_col: str, close_col: str) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    df["__date_key"] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
    merged = df.merge(regime, left_on="__date_key", right_index=True, how="left")

    close = pd.to_numeric(merged[close_col], errors="coerce")
    stock_return_20d = close.pct_change(20)
    stock_log_return = np.log(close).diff()

    nan_series = pd.Series(np.nan, index=merged.index)
    nifty_return_20d = merged["nifty_return_20d"] if "nifty_return_20d" in merged.columns else nan_series
    nifty_log_return = merged["nifty_log_return"] if "nifty_log_return" in merged.columns else nan_series

    merged["return_20d"] = stock_return_20d
    merged["rel_strength_20d"] = stock_return_20d - nifty_return_20d

    # Rolling 60-day beta vs NIFTY (covariance / variance)
    if nifty_log_return.notna().any():
        rolling_cov = stock_log_return.rolling(60).cov(nifty_log_return)
        rolling_var = nifty_log_return.rolling(60).var()
        merged["beta_60d"] = rolling_cov / rolling_var.replace(0, np.nan)
    else:
        merged["beta_60d"] = np.nan

    # Calendar features
    dates = merged["__date_key"]
    merged["day_of_week"] = dates.dt.dayofweek
    merged["day_of_month"] = dates.dt.day
    month_end = dates + pd.offsets.MonthEnd(0)
    merged["days_to_month_end"] = (month_end - dates).dt.days

    merged = merged.drop(columns=["__date_key"])
    return merged


def _add_cross_sectional_ranks(
    sheets: Dict[str, pd.DataFrame],
    date_col: str,
    rank_cols: Iterable[str],
) -> Dict[str, pd.DataFrame]:
    """For each rank_col, percentile-rank across the universe on each date.

    Universe = all tickers in ``sheets`` that have a value on that date.
    """
    if not sheets:
        return sheets

    frames: List[pd.DataFrame] = []
    for ticker, df in sheets.items():
        if df is None or df.empty:
            continue
        slim = df[[date_col, *(c for c in rank_cols if c in df.columns)]].copy()
        slim["__ticker"] = ticker
        slim["__date_key"] = pd.to_datetime(slim[date_col]).dt.tz_localize(None).dt.normalize()
        frames.append(slim)

    if not frames:
        return sheets
    universe = pd.concat(frames, ignore_index=True)

    rank_outputs: Dict[str, pd.DataFrame] = {}
    for col in rank_cols:
        if col not in universe.columns:
            continue
        ranked = (
            universe.groupby("__date_key")[col]
            .rank(pct=True, method="average")
            .rename(f"{col}_rank")
        )
        rank_outputs[col] = pd.concat([universe[["__ticker", "__date_key"]], ranked], axis=1)

    if not rank_outputs:
        return sheets

    out: Dict[str, pd.DataFrame] = {}
    for ticker, df in sheets.items():
        df = df.copy()
        df["__date_key"] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
        for col, ranked in rank_outputs.items():
            ticker_rank = ranked[ranked["__ticker"] == ticker][["__date_key", f"{col}_rank"]]
            df = df.merge(ticker_rank, on="__date_key", how="left")
        df = df.drop(columns=["__date_key"])
        out[ticker] = df
    return out


def add_cross_sectional_features(
    sheets: Dict[str, pd.DataFrame],
    date_col: str = "Date",
    close_col: str = "Close",
    index_df: Optional[pd.DataFrame] = None,
    rank_cols: Iterable[str] = ("RSI_14", "MACD_Histogram", "ATR_14", "ROC_12"),
    fundamentals_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Add regime + relative-strength + rank + earnings-event features."""
    index_df = index_df if index_df is not None else load_index_history()
    if index_df.empty:
        _log("[cross-sectional] no index history available; regime features will be NaN")
        regime = pd.DataFrame()
    else:
        regime = _compute_regime(index_df)

    enriched: Dict[str, pd.DataFrame] = {}
    for ticker, df in sheets.items():
        if df is None or df.empty or close_col not in df.columns or date_col not in df.columns:
            enriched[ticker] = df
            continue
        enriched[ticker] = _attach_per_stock(df, regime, date_col=date_col, close_col=close_col)

    enriched = _add_cross_sectional_ranks(enriched, date_col=date_col, rank_cols=rank_cols)

    if fundamentals_df is not None and not fundamentals_df.empty:
        f = fundamentals_df.set_index("ticker")
        for ticker, df in enriched.items():
            if df is None or df.empty or ticker not in f.index:
                continue
            row = f.loc[ticker]
            days_to_earn = row.get("days_to_earnings")
            in_window = row.get("in_earnings_window_5d")
            df["days_to_earnings"] = days_to_earn
            df["in_earnings_window_5d"] = in_window
            enriched[ticker] = df

    return enriched


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh NIFTY/VIX index history cache.")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=str(DEFAULT_INDEX_CACHE))
    args = parser.parse_args(argv)

    df = fetch_index_history(start=args.start, end=args.end, output_path=Path(args.output))
    print(json.dumps({"rows": int(len(df))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
