"""Daily production run — refresh OHLCV, engineer features, book target/stop exits.

Cadence: **every trading day.** This is the production analogue of the backtest's
per-bar exit monitor (``backtesting.exits``): it does NOT retrain, forecast, or
call the LLM. It keeps the price data current and then checks whether any *held*
position touched its LLM target (``sell_price``) or stoploss today — booking the
exit against the live holdings book if so.

Stages:
  1. OHLCV update            (ingestion.collect_all, market-data step only) — subprocess
  2. Feature engineering     (Feature_Engineering.compute_indicators → ATR14 for slippage)
  3. Target/stop exit check  (backtesting.exits.scan_exits on today's O/H/L/C vs the
                              stored trade_suggestions levels) → book fills, update
                              state/portfolio_holdings.json, write a daily report

Holdings live in ``state/portfolio_holdings.json`` (the same book the production
allocation layer writes). Suggestions (with target + stoploss) are read from
Firestore ``trade_suggestions``. A held name with no active suggestion is simply
held (no forced exit).

Usage::

    python -m scripts.run_daily
    python -m scripts.run_daily --skip-ohlcv       # only run the exit check
    python -m scripts.run_daily --dry-run          # print actions, don't write state
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._pipeline import run_stage  # noqa: E402


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _load_suggestions() -> Dict[str, Dict[str, Any]]:
    """Standing LLM suggestions (target/stop levels) from Firestore, keyed by ticker."""
    from features.trade_suggestions import SUGGESTIONS_COLLECTION
    from ingestion._firestore import init_firestore_client
    client = init_firestore_client()
    out: Dict[str, Dict[str, Any]] = {}
    for doc in client.collection(SUGGESTIONS_COLLECTION).stream():
        data = doc.to_dict() or {}
        ticker = str(data.get("ticker") or doc.id).upper()
        out[ticker] = {
            "action": data.get("action"),
            "sell_price": data.get("sell_price"),
            "stoploss": data.get("stoploss"),
            "buy_price": data.get("buy_price"),
        }
    return out


def _latest_ohlc_and_atr(ticker: str, *, lookback: str = "3mo") -> Optional[Dict[str, float]]:
    """Today's O/H/L/C for ``ticker`` plus ATR14 (the feature-engineering step)."""
    import yfinance as yf
    from backtesting.costs import atr14_from_frame
    try:
        raw = yf.download(f"{ticker}.NS", period=lookback, interval="1d",
                          progress=False, auto_adjust=True)
    except Exception as exc:  # noqa: BLE001
        _log(f"[daily] {ticker}: OHLC fetch failed ({exc})")
        return None
    if raw is None or raw.empty:
        return None
    if hasattr(raw.columns, "levels"):
        raw.columns = [c[0] for c in raw.columns]
    # Feature engineering: compute the full indicator set (best-effort), and derive
    # ATR14 for the slippage model. atr14_from_frame is the robust primitive.
    try:
        from Feature_Engineering import compute_indicators
        compute_indicators(raw.copy())
    except Exception:  # pragma: no cover - FE is non-fatal for the price check
        pass
    last = raw.iloc[-1]
    try:
        return {
            "open": float(last["Open"]), "high": float(last["High"]),
            "low": float(last["Low"]), "close": float(last["Close"]),
            "atr14": float(atr14_from_frame(raw)),
        }
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Exit check + book update
# ---------------------------------------------------------------------------

def check_exits(*, dry_run: bool = False) -> Dict[str, Any]:
    """Book any target/stop exits for held positions using today's prices."""
    from backtesting import exits as bt_exits
    from backtesting import portfolio as bt_portfolio
    from features.portfolio_allocation import DEFAULT_HOLDINGS_STATE, Holdings

    capital = float(os.environ.get("ALLOC_INITIAL_CAPITAL", "1000000"))
    cost_bps = float(os.environ.get("DAILY_COST_BPS_ROUNDTRIP", "40"))
    slip_mult = float(os.environ.get("DAILY_SLIPPAGE_ATR_MULT", "0.10"))
    honor_stop = os.environ.get("DAILY_HONOR_STOPLOSS", "true").strip().lower() in {"1", "true", "yes", "on"}
    state_path = Path(os.environ.get("PORTFOLIO_HOLDINGS_STATE", str(DEFAULT_HOLDINGS_STATE)))

    holdings = Holdings.load(state_path, capital=capital)
    held = {t.upper(): p for t, p in holdings.positions.items() if float(p.get("qty", 0)) > 0}
    if not held:
        _log("[daily] no open positions — nothing to monitor")
        return {"status": "ok", "held": 0, "exits": [], "monitor": []}

    signals = _load_suggestions()
    ohlc: Dict[str, Dict[str, float]] = {}
    atr_map: Dict[str, float] = {}
    for t in held:
        bar = _latest_ohlc_and_atr(t)
        if bar:
            ohlc[t] = {k: bar[k] for k in ("open", "high", "low", "close")}
            atr_map[t] = bar["atr14"]

    exits = bt_exits.scan_exits(held, signals, ohlc, honor_stop=honor_stop)
    monitor = bt_exits.monitor_rows(datetime.now(timezone.utc).date().isoformat(),
                                    held, signals, ohlc, exits, honor_stop=honor_stop)

    booked: List[Dict[str, Any]] = []
    if exits:
        fills = bt_portfolio.fill_exits(exits, held, atr_map=atr_map,
                                        cost_bps=cost_bps, slippage_atr_mult=slip_mult)
        by_ticker = {e.ticker.upper(): e for e in exits}
        for f in fills:
            t = f.ticker.upper()
            pos = holdings.positions.get(t)
            if not pos:
                continue
            qty = min(float(f.qty), float(pos.get("qty", 0.0)))
            avg = float(pos.get("avg_price", 0.0))
            realized = (float(f.price) - avg) * qty - float(f.cost)
            ex = by_ticker.get(t)
            booked.append({
                "ticker": t, "reason": getattr(ex, "reason", f.reason),
                "qty": round(qty, 4), "fill_price": round(float(f.price), 4),
                "proceeds": round(float(f.gross), 2), "cost": round(float(f.cost), 2),
                "realized_pnl": round(realized, 2),
                "target": getattr(ex, "target", None), "stoploss": getattr(ex, "stop", None),
            })
            if not dry_run:
                holdings.cash += float(f.cash_delta)
                pos["qty"] = float(pos.get("qty", 0.0)) - qty
                if pos["qty"] <= 1e-9:
                    holdings.positions.pop(t, None)

    if not dry_run:
        _save_holdings(holdings, state_path)
    _write_report(monitor, booked)

    for b in booked:
        _log(f"[daily] EXIT {b['ticker']} {b['reason']} @ ₹{b['fill_price']} "
             f"pnl ₹{b['realized_pnl']:,.0f}")
    return {"status": "ok", "held": len(held), "exits": booked, "monitor": monitor,
            "cash": round(holdings.cash, 2)}


def _save_holdings(holdings: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(
        {"capital": holdings.capital, "cash": holdings.cash, "positions": holdings.positions},
        indent=2, default=str))
    os.replace(tmp, path)


def _write_report(monitor: List[Dict[str, Any]], exits: List[Dict[str, Any]]) -> None:
    try:
        import pandas as pd
        out = Path(os.environ.get("DAILY_MONITOR_REPORT", "outputs/production_daily_monitor.xlsx"))
        out.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out, engine="openpyxl") as xl:
            (pd.DataFrame(monitor) if monitor else pd.DataFrame({"info": ["no positions monitored"]})
             ).to_excel(xl, sheet_name="Price_Monitor", index=False)
            (pd.DataFrame(exits) if exits else pd.DataFrame({"info": ["no exits today"]})
             ).to_excel(xl, sheet_name="Exits", index=False)
        _log(f"[daily] wrote {out}")
    except Exception as exc:  # noqa: BLE001 - reporting is best-effort
        _log(f"[daily] report skipped: {exc}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Daily: OHLCV → feature engineering → target/stop exit check.")
    p.add_argument("--tickers", default=None, help="Comma-separated subset for the OHLCV step.")
    p.add_argument("--skip-ohlcv", action="store_true", help="Skip the OHLCV update; only run the exit check.")
    p.add_argument("--dry-run", action="store_true", help="Report exits without writing holdings/state.")
    args = p.parse_args(argv)

    stages: List[Dict[str, Any]] = []
    if not args.skip_ohlcv:
        ohlcv_cmd = ["-m", "ingestion.collect_all", "--no-cross-sectional",
                     "--no-news", "--no-reddit", "--no-x", "--no-sentiment"]
        if args.tickers:
            ohlcv_cmd += ["--tickers", args.tickers]
        stages.append(run_stage("ohlcv_update", ohlcv_cmd, dry_run=args.dry_run))

    exit_result = check_exits(dry_run=args.dry_run)
    stages.append({"stage": "exit_check", **exit_result})

    failed = [s for s in stages if s.get("status") == "error"]
    summary = {"pipeline": "daily", "status": "error" if failed else "ok", "stages": stages}
    print(json.dumps(summary, indent=2, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
