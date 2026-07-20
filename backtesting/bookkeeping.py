"""Local Excel bookkeeping for the walk-forward backtest.

Every artefact the simulation produces is recorded to a single ``.xlsx`` workbook
so the whole run is auditable offline, one sheet per record type:

* ``LLM_Suggestions`` — every GLM BUY/HOLD/AVOID call (price levels, stop,
  confidence, rationale, sentiment/fundamentals provenance).
* ``Allocations``     — the quantitative sizing our engine produced for each
  rebalance (weight, target ₹, μ, σ, cap, risk-reward).
* ``Reallocations``   — cash-guard capital rotations (pulled ₹ out of A to fund B).
* ``Price_Monitor``   — the daily price check per held name with an active target:
  that bar's O/H/L/C, the target/stop levels, and whether either was hit.
* ``Exits``           — target/stop profit-bookings: reason, trigger vs fill price,
  quantity and **sale amount** freed back to cash.
* ``Trades``          — every fill: side, qty, price, gross, cost, **sale amount**,
  realised P&L, resulting cash.
* ``EquityCurve``     — per-bar cash / positions value / equity vs benchmark.
* ``PnL_Statement``   — the final profit-and-loss: realised + unrealised + costs,
  total return, and a per-ticker breakdown.
* ``Open_Positions``  — what is still held at the end, with mark and unrealised P&L.

The bookkeeper accumulates rows in memory during the run and writes the workbook
atomically on :meth:`flush` (called every checkpoint so a crash keeps the latest).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class BacktestBookkeeper:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.suggestions: List[Dict[str, Any]] = []
        self.allocations: List[Dict[str, Any]] = []
        self.reallocations: List[Dict[str, Any]] = []

    # ── recorders ────────────────────────────────────────────────────────────
    def log_suggestions(self, bar: Any, signals: Dict[str, Dict[str, Any]]) -> None:
        b = _iso(bar)
        for ticker, s in signals.items():
            self.suggestions.append({
                "bar": b,
                "ticker": ticker.upper(),
                "action": s.get("action"),
                "buy_price": s.get("buy_price"),
                "sell_day": s.get("sell_day"),
                "sell_price": s.get("sell_price"),
                "stoploss": s.get("stoploss"),
                "confidence": s.get("confidence"),
                "sentiment_used": s.get("sentiment_used"),
                "fundamentals_used": s.get("fundamentals_used"),
                "review_decision": s.get("review_decision"),
                "gate_reasons": "; ".join(s.get("gate_reasons") or []) or None,
                "rationale": s.get("rationale"),
            })

    def log_allocation(
        self,
        bar: Any,
        opt_result: Any,
        equity: float,
        buy_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        b = _iso(bar)
        buy_meta = buy_meta or {}
        deploy_frac = float(getattr(opt_result, "deploy_fraction", float("nan")))
        weighted_rr = float(getattr(opt_result, "weighted_risk_reward", float("nan")))
        for ticker, weight in opt_result.weights.items():
            meta = buy_meta.get(ticker.upper())
            self.allocations.append({
                "bar": b,
                "ticker": ticker.upper(),
                "weight": round(float(weight), 6),
                "target_notional": round(float(weight) * float(equity), 2),
                "deploy_fraction": round(deploy_frac, 4),
                "portfolio_risk_reward": round(weighted_rr, 4) if weighted_rr == weighted_rr else None,
                "expected_return_15d": round(float(opt_result.expected_return.get(ticker, float("nan"))), 6),
                "sigma_annual": round(float(opt_result.sigma_annual.get(ticker, float("nan"))), 6),
                "cap": round(float(opt_result.cap.get(ticker, float("nan"))), 6),
                "risk_reward": round(float(getattr(meta, "risk_reward", float("nan"))), 4) if meta else None,
                "confidence": round(float(getattr(meta, "confidence", float("nan"))), 4) if meta else None,
            })

    def log_reallocations(self, bar: Any, reconcile_result: Any) -> None:
        b = _iso(bar)
        for r in getattr(reconcile_result, "reallocations", []) or []:
            row = {"bar": b}
            row.update(r)
            self.reallocations.append(row)

    # ── final write ──────────────────────────────────────────────────────────
    def flush(self, state: Any) -> Path:
        """Write the full workbook from accumulated rows + the live state logs."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sheets: Dict[str, pd.DataFrame] = {
            "LLM_Suggestions": _df(self.suggestions),
            "Allocations": _df(self.allocations),
            "Reallocations": _df(self.reallocations),
            "Price_Monitor": _df(getattr(state, "price_monitor_log", [])),
            "Exits": _df(getattr(state, "exit_log", [])),
            "Trades": _df(state.trade_log),
            "EquityCurve": _df(state.equity_curve),
            "Forecasts": _df(state.forecast_log),
            "PnL_Statement": _pnl_statement(state),
            "PnL_Ex_Costs": _pnl_statement(state, exclude_costs=True),
            "Open_Positions": _open_positions(state),
        }
        tmp = self.path.with_suffix(".xlsx.tmp")
        with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
            for name, df in sheets.items():
                (df if not df.empty else pd.DataFrame({"info": [f"no {name} rows"]})).to_excel(
                    writer, sheet_name=name[:31], index=False
                )
        import os
        os.replace(tmp, self.path)
        return self.path


def _df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _open_positions(state: Any) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ticker, pos in state.positions.items():
        qty = float(pos.get("qty", 0.0))
        if qty <= 0:
            continue
        avg = float(pos.get("avg_price", 0.0))
        mark = float(state.mark_prices.get(ticker.upper(), avg) or avg)
        rows.append({
            "ticker": ticker.upper(),
            "qty": round(qty, 4),
            "avg_price": round(avg, 4),
            "mark_price": round(mark, 4),
            "cost_basis": round(qty * avg, 2),
            "market_value": round(qty * mark, 2),
            "unrealized_pnl": round((mark - avg) * qty, 2),
        })
    return pd.DataFrame(rows)


def _pnl_statement(state: Any, *, exclude_costs: bool = False) -> pd.DataFrame:
    """The P&L statement; ``exclude_costs=True`` adds every rupee of transaction
    cost back to equity/realized so the sheet answers "what would the strategy
    have earned with free execution?" (the cost drag isolated in one number)."""
    positions_value = state.position_value()
    unrealized = sum(
        (float(state.mark_prices.get(t.upper(), p.get("avg_price", 0.0)) or 0.0) - float(p.get("avg_price", 0.0)))
        * float(p.get("qty", 0.0))
        for t, p in state.positions.items()
    )
    equity = state.equity()
    costs = float(state.total_costs)
    realized = float(state.realized_pnl)
    if exclude_costs:
        equity += costs
        realized += costs
    total_return = equity - state.initial_capital
    rows = [
        ("Initial capital", round(state.initial_capital, 2)),
        ("Ending cash", round(state.cash + (costs if exclude_costs else 0.0), 2)),
        ("Ending positions value", round(positions_value, 2)),
        ("Ending equity", round(equity, 2)),
        ("Realized P&L", round(realized, 2)),
        ("Unrealized P&L", round(unrealized, 2)),
        ("Total transaction costs", 0.0 if exclude_costs else round(costs, 2)),
        ("Net total return (₹)", round(total_return, 2)),
        ("Net total return (%)", round(100.0 * total_return / state.initial_capital, 4) if state.initial_capital else 0.0),
        ("Trades executed", len(state.trade_log)),
        ("Costs excluded", exclude_costs),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)
