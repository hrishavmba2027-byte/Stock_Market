"""Common result sheet for a completed backtest — benchmarked vs NIFTY 50,
Sensex and government bonds.

Reads the run's ``state`` checkpoint (the equity curve the harness recorded) and
builds a single, shareable Excel workbook plus a chart comparing the strategy to
three benchmarks, all rebased to the same ₹1-crore starting capital:

* **NIFTY 50** (``^NSEI``) and **Sensex** (``^BSESN``) — total-price series pulled
  via yfinance for the run window (NIFTY falls back to the local index cache).
* **Government bonds** — a buy-and-hold G-Sec proxy: a constant annual yield
  (default 7.0%, configurable) compounded across the run's trading days. A real
  bond-ETF symbol can be supplied instead via ``--bond-symbol``.

Metrics per track (reusing :mod:`backtesting.evaluate`): ending value, total
return, CAGR, annualised volatility, Sharpe, max drawdown — plus the strategy's
alpha/beta/up-down capture against each index.

Output workbook (default ``outputs/backtest/results.xlsx``) sheets:
  ``Summary``  — the comparison table (one row per track)
  ``Equity``   — daily aligned equity of every track, rebased to ₹1 cr
  ``Benchmark_Stats`` — alpha/beta/capture of the strategy vs each index

Usage::

    python -m backtesting.results
    python -m backtesting.results --bond-yield 0.072 --out outputs/backtest/results.xlsx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.backtest_settings import get_backtest_settings
from backtesting import evaluate


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_strategy_equity(state_path: Path) -> pd.Series:
    """Strategy equity as a date-indexed ₹ series from the state checkpoint."""
    data = json.loads(Path(state_path).read_text())
    curve = data.get("equity_curve") or []
    if not curve:
        raise SystemExit(f"[results] no equity_curve in {state_path} — run the simulation first.")
    idx = pd.to_datetime([r["bar"] for r in curve])
    return pd.Series([float(r["equity"]) for r in curve], index=idx).sort_index()


def _yf_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                         end=(end + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                         interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception as exc:  # noqa: BLE001
        _log(f"[results] yfinance fetch failed for {symbol}: {exc}")
        return None


def _nifty_from_cache(index_cache: Path, symbol: str) -> Optional[pd.Series]:
    try:
        from features.cross_sectional import load_index_history
        df = load_index_history(Path(index_cache))
        if df is None or df.empty:
            return None
        col = symbol if symbol in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        return None


def gov_bond_series(dates: pd.DatetimeIndex, annual_yield: float, initial: float) -> pd.Series:
    """Buy-and-hold G-Sec proxy: ``initial`` compounding at ``annual_yield`` p.a.

    Compounds per trading day (``(1+y)^(1/252)``) across the run's own bar dates so
    it aligns exactly with the strategy calendar. This is a deterministic,
    offline-safe fixed-income baseline; pass ``--bond-symbol`` for a real ETF.
    """
    daily = (1.0 + annual_yield) ** (1.0 / evaluate.TRADING_DAYS)
    steps = np.arange(len(dates))
    return pd.Series(initial * daily ** steps, index=dates)


def _rebase(series: pd.Series, dates: pd.DatetimeIndex, initial: float) -> Optional[pd.Series]:
    """Align a benchmark price series to the strategy dates and rebase to ``initial``."""
    if series is None or series.empty:
        return None
    s = series[~series.index.duplicated(keep="last")].sort_index()
    aligned = s.reindex(dates.union(s.index)).sort_index().ffill().reindex(dates).ffill().bfill()
    if aligned.dropna().empty or aligned.iloc[0] == 0:
        return None
    return aligned / aligned.iloc[0] * initial


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _curve(series: pd.Series) -> List[Dict[str, Any]]:
    return [{"bar": ts.strftime("%Y-%m-%d"), "equity": float(v)} for ts, v in series.items()]


def _track_row(name: str, series: pd.Series, initial: float, rf: float) -> Dict[str, Any]:
    m = evaluate.equity_metrics(_curve(series), rf_annual=rf)
    return {
        "track": name,
        "start_value": round(initial, 2),
        "ending_value": round(float(series.iloc[-1]), 2),
        "total_return_pct": round(m["total_return"] * 100, 2),
        "cagr_pct": round(m["cagr"] * 100, 2),
        "volatility_annual_pct": round(m["volatility_annual"] * 100, 2),
        "sharpe": round(m["sharpe"], 3),
        "sortino": round(m["sortino"], 3),
        "max_drawdown_pct": round(m["max_drawdown"] * 100, 2),
        "calmar": round(m["calmar"], 3),
    }


def build_results(
    state_path: Path,
    *,
    initial_capital: float,
    nifty_symbol: str = "^NSEI",
    sensex_symbol: str = "^BSESN",
    bond_yield: float = 0.07,
    bond_symbol: Optional[str] = None,
    index_cache: Optional[Path] = None,
    rf: float = 0.0,
) -> Dict[str, Any]:
    strat = load_strategy_equity(state_path)
    dates = strat.index
    start, end = dates[0], dates[-1]
    _log(f"[results] strategy: {len(strat)} bars {start.date()}..{end.date()} "
         f"ending ₹{strat.iloc[-1]:,.0f}")

    tracks: Dict[str, pd.Series] = {"Strategy": strat}

    nifty = _yf_close(nifty_symbol, start, end)
    if nifty is None and index_cache is not None:
        nifty = _nifty_from_cache(index_cache, nifty_symbol)
    nifty_r = _rebase(nifty, dates, initial_capital)
    if nifty_r is not None:
        tracks["NIFTY 50"] = nifty_r
    else:
        _log("[results] NIFTY 50 unavailable — skipped")

    sensex_r = _rebase(_yf_close(sensex_symbol, start, end), dates, initial_capital)
    if sensex_r is not None:
        tracks["Sensex"] = sensex_r
    else:
        _log("[results] Sensex unavailable — skipped")

    if bond_symbol:
        bonds_r = _rebase(_yf_close(bond_symbol, start, end), dates, initial_capital)
        bond_label = f"Gov Bonds ({bond_symbol})"
    else:
        bonds_r = gov_bond_series(dates, bond_yield, initial_capital)
        bond_label = f"Gov Bonds ({bond_yield*100:.1f}% p.a.)"
    if bonds_r is not None:
        tracks[bond_label] = bonds_r

    summary = [_track_row(name, s, initial_capital, rf) for name, s in tracks.items()]

    bench_stats = []
    for name in ("NIFTY 50", "Sensex"):
        if name in tracks:
            bc = evaluate.benchmark_compare(_curve(strat), tracks[name])
            bench_stats.append({"benchmark": name, **{k: round(v, 4) for k, v in bc.items()}})

    equity_df = pd.DataFrame({name: s for name, s in tracks.items()})
    equity_df.index.name = "date"

    return {
        "summary": summary,
        "benchmark_stats": bench_stats,
        "equity_df": equity_df,
        "window": {"start": str(start.date()), "end": str(end.date()), "bars": len(strat)},
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(results: Dict[str, Any], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(results["summary"])
    bench_df = pd.DataFrame(results["benchmark_stats"])
    equity_df = results["equity_df"].round(2)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        summary_df.to_excel(xl, sheet_name="Summary", index=False)
        if not bench_df.empty:
            bench_df.to_excel(xl, sheet_name="Benchmark_Stats", index=False)
        equity_df.to_excel(xl, sheet_name="Equity")
    _log(f"[results] wrote {out_path}")

    try:
        _write_chart(equity_df, out_path.with_name("results_equity_vs_benchmarks.png"))
    except Exception as exc:  # pragma: no cover - plotting optional
        _log(f"[results] chart skipped: {exc}")


def _write_chart(equity_df: pd.DataFrame, png_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for col in equity_df.columns:
        ax.plot(equity_df.index, equity_df[col], label=col, linewidth=1.6 if col == "Strategy" else 1.0)
    ax.set_title("Backtest — Strategy vs NIFTY 50, Sensex & Government Bonds (₹, rebased to start)")
    ax.set_ylabel("Portfolio value (₹)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    _log(f"[results] wrote {png_path}")


def run(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    settings = get_backtest_settings()
    p = argparse.ArgumentParser(description="Backtest result sheet vs NIFTY 50, Sensex & gov bonds.")
    p.add_argument("--state", default=str(settings.state_path), help="Run state checkpoint (equity curve source).")
    p.add_argument("--out", default=str(Path(settings.report_dir).parent / "results.xlsx"))
    p.add_argument("--nifty-symbol", default="^NSEI")
    p.add_argument("--sensex-symbol", default="^BSESN")
    p.add_argument("--bond-yield", type=float, default=0.07, help="Annual G-Sec yield for the bond proxy (default 0.07).")
    p.add_argument("--bond-symbol", default=None, help="Use a real bond-ETF symbol instead of the yield proxy.")
    p.add_argument("--rf", type=float, default=0.0, help="Risk-free rate for Sharpe/Sortino (annual).")
    args = p.parse_args(argv)

    results = build_results(
        Path(args.state), initial_capital=settings.initial_capital,
        nifty_symbol=args.nifty_symbol, sensex_symbol=args.sensex_symbol,
        bond_yield=args.bond_yield, bond_symbol=args.bond_symbol,
        index_cache=Path(settings.index_cache_path), rf=args.rf,
    )
    write_results(results, Path(args.out))
    print(json.dumps({"window": results["window"], "summary": results["summary"],
                      "benchmark_stats": results["benchmark_stats"]}, indent=2, default=str))
    return results


if __name__ == "__main__":
    run()
