"""``BacktestState`` — the simulated book, checkpointed for a resumable run.

Holds cash, positions (with average cost basis for realised-P&L), the last
forecast/signal/sentiment fingerprints, the two-deep pending-order queue that
enforces next-bar fills, and the equity/trade/forecast/signal logs. Every field
round-trips through JSON so a multi-year run can resume exactly where it stopped.

Invariant: applying fills can **never** drive ``cash`` below zero — the
reconciler upstream only ever proposes BUYs it can fund, and a defensive clamp
here refuses any fill that would overdraw the book.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class BacktestState:
    cash: float
    initial_capital: float
    positions: Dict[str, Dict[str, float]] = field(default_factory=dict)   # ticker -> {qty, avg_price}
    mark_prices: Dict[str, float] = field(default_factory=dict)
    last_forecast: Dict[str, Dict[str, float]] = field(default_factory=dict)
    last_signal: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_sentiment_fp: Dict[str, str] = field(default_factory=dict)
    pending_orders: List[Dict[str, Any]] = field(default_factory=list)
    pending_orders_prev: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    forecast_log: List[Dict[str, Any]] = field(default_factory=list)
    signal_log: List[Dict[str, Any]] = field(default_factory=list)
    price_monitor_log: List[Dict[str, Any]] = field(default_factory=list)  # daily target/stop watch
    exit_log: List[Dict[str, Any]] = field(default_factory=list)           # booked target/stop exits
    realized_pnl: float = 0.0
    total_costs: float = 0.0
    models_cutoff: Optional[str] = None          # last retrain date (None => unseeded)
    last_bar: Optional[str] = None

    # ── marking ────────────────────────────────────────────────────────────
    def mark_to_market(self, closes: Dict[str, float]) -> None:
        for t, px in closes.items():
            if px is not None and px > 0:
                self.mark_prices[t.upper()] = float(px)

    def position_value(self) -> float:
        total = 0.0
        for t, pos in self.positions.items():
            qty = float(pos.get("qty", 0.0))
            price = float(self.mark_prices.get(t.upper(), pos.get("avg_price", 0.0)) or 0.0)
            total += qty * price
        return total

    def equity(self) -> float:
        return float(self.cash) + self.position_value()

    # ── fills ────────────────────────────────────────────────────────────────
    def apply_fills(self, fills: List[Any], *, bar: Any = None) -> None:
        """Apply next-bar fills to cash + positions, recording realised P&L.

        A BUY that cannot be funded from cash is refused (defensive; the
        reconciler should never emit one). SELLs realise P&L against the average
        cost basis and free the shares.
        """
        bar_str = _iso(bar)
        for f in fills:
            ticker = f.ticker.upper()
            qty = float(f.qty)
            price = float(f.price)
            cost = float(getattr(f, "cost", 0.0))
            if f.side == "BUY":
                if f.cash_delta < -self.cash - 1e-6:
                    # Never overdraw. Skip (should not happen post-reconcile).
                    continue
                self.cash += f.cash_delta
                self._add_shares(ticker, qty, price, bar=bar_str)
                realized = 0.0
            else:  # SELL
                self.cash += f.cash_delta
                realized = self._remove_shares(ticker, qty, price) - cost
                self.realized_pnl += realized
            self.total_costs += cost
            self.trade_log.append({
                "bar": bar_str, "ticker": ticker, "side": f.side, "qty": round(qty, 4),
                "price": round(price, 4), "gross": round(f.gross, 2), "cost": round(cost, 2),
                "cash_delta": round(f.cash_delta, 2), "realized_pnl": round(realized, 2),
                "cash_after": round(self.cash, 2), "reason": getattr(f, "reason", ""),
            })
        self.cash = max(0.0, self.cash)

    def _add_shares(self, ticker: str, qty: float, price: float, *, bar: Any = None) -> None:
        pos = self.positions.setdefault(ticker, {"qty": 0.0, "avg_price": 0.0})
        old_qty = float(pos["qty"])
        new_qty = old_qty + qty
        if new_qty <= 0:
            self.positions.pop(ticker, None)
            return
        pos["avg_price"] = (old_qty * float(pos["avg_price"]) + qty * price) / new_qty
        pos["qty"] = new_qty
        # First lot of a fresh episode stamps the entry bar (drives the age exit
        # and the position-review payload); top-ups keep the original date.
        if old_qty <= 1e-9 or not pos.get("entry_bar"):
            pos["entry_bar"] = bar

    def _remove_shares(self, ticker: str, qty: float, price: float) -> float:
        pos = self.positions.get(ticker)
        if not pos:
            return 0.0
        avg = float(pos["avg_price"])
        sell_qty = min(qty, float(pos["qty"]))
        realized = (price - avg) * sell_qty
        pos["qty"] = float(pos["qty"]) - sell_qty
        if pos["qty"] <= 1e-9:
            self.positions.pop(ticker, None)
        return realized

    # ── target/stop exits ────────────────────────────────────────────────────
    def mark_exited(self, ticker: str) -> None:
        """Retire a name's standing suggestion after a target/stop exit.

        Dropping it from ``last_signal`` means the rebalance step no longer treats
        the name as an actionable BUY/HOLD, so the freed cash is held until the
        framework issues a *fresh* suggestion (the next retrain, or a sentiment
        change). Any stale mark is left in place for equity accounting only.
        """
        self.last_signal.pop(ticker.upper(), None)
        self.last_signal.pop(ticker, None)

    def record_exit(self, bar: Any, exit_signal: Any, fill: Any) -> None:
        self.exit_log.append({
            "bar": _iso(bar),
            "ticker": exit_signal.ticker.upper(),
            "reason": exit_signal.reason,
            "target": exit_signal.target,
            "stoploss": exit_signal.stop,
            "trigger_price": round(float(exit_signal.price), 4),
            "fill_price": round(float(fill.price), 4),
            "qty": round(float(fill.qty), 4),
            "sale_amount": round(float(fill.gross), 2),
            "cost": round(float(fill.cost), 2),
        })

    def record_price_monitor(self, rows: List[Dict[str, Any]]) -> None:
        self.price_monitor_log.extend(rows)

    # ── logging ────────────────────────────────────────────────────────────
    def record_equity(self, bar: Any, *, benchmark: Optional[float] = None) -> None:
        self.last_bar = _iso(bar)
        self.equity_curve.append({
            "bar": self.last_bar,
            "cash": round(self.cash, 2),
            "positions_value": round(self.position_value(), 2),
            "equity": round(self.equity(), 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "total_costs": round(self.total_costs, 2),
            "n_positions": len([p for p in self.positions.values() if p.get("qty", 0) > 0]),
            "benchmark": benchmark,
        })

    # ── persistence ──────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cash": self.cash,
            "initial_capital": self.initial_capital,
            "positions": self.positions,
            "mark_prices": self.mark_prices,
            "last_forecast": self.last_forecast,
            "last_signal": self.last_signal,
            "last_sentiment_fp": self.last_sentiment_fp,
            "pending_orders": self.pending_orders,
            "pending_orders_prev": self.pending_orders_prev,
            "equity_curve": self.equity_curve,
            "trade_log": self.trade_log,
            "forecast_log": self.forecast_log,
            "signal_log": self.signal_log,
            "price_monitor_log": self.price_monitor_log,
            "exit_log": self.exit_log,
            "realized_pnl": self.realized_pnl,
            "total_costs": self.total_costs,
            "models_cutoff": self.models_cutoff,
            "last_bar": self.last_bar,
        }

    def checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), default=str, indent=2))
        os.replace(tmp, path)

    @classmethod
    def load_or_init(cls, path: Path, *, initial_capital: float, resume: bool = True) -> "BacktestState":
        path = Path(path)
        if resume and path.exists():
            data = json.loads(path.read_text())
            return cls(
                cash=float(data.get("cash", initial_capital)),
                initial_capital=float(data.get("initial_capital", initial_capital)),
                positions=data.get("positions", {}),
                mark_prices=data.get("mark_prices", {}),
                last_forecast=data.get("last_forecast", {}),
                last_signal=data.get("last_signal", {}),
                last_sentiment_fp=data.get("last_sentiment_fp", {}),
                pending_orders=data.get("pending_orders", []),
                pending_orders_prev=data.get("pending_orders_prev", []),
                equity_curve=data.get("equity_curve", []),
                trade_log=data.get("trade_log", []),
                forecast_log=data.get("forecast_log", []),
                signal_log=data.get("signal_log", []),
                price_monitor_log=data.get("price_monitor_log", []),
                exit_log=data.get("exit_log", []),
                realized_pnl=float(data.get("realized_pnl", 0.0)),
                total_costs=float(data.get("total_costs", 0.0)),
                models_cutoff=data.get("models_cutoff"),
                last_bar=data.get("last_bar"),
            )
        return cls(cash=float(initial_capital), initial_capital=float(initial_capital))


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)
