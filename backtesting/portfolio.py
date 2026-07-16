"""Backtest portfolio layer — thin adapter over :mod:`allocation.engine`.

The quantitative sizing/reconciliation math lives in the shared engine (so the
backtest and production are provably identical). This module wires that engine to
the harness: it derives μ from the forecast path, calls the optimiser, turns
target weights into orders through the cash-guard reconciler, and settles orders
at the next bar's open with transaction cost + ATR slippage applied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from allocation.config import AllocationConfig
from allocation.engine import (
    BuyMeta,
    OptimizeResult,
    PositionView,
    ReconcileResult,
    optimize_weights,
    reconcile_orders,
    risk_reward_ratio,
)
from backtesting import costs as costs_mod

FORECAST_HORIZON = "T+15"


@dataclass
class Fill:
    ticker: str
    side: str            # "BUY" | "SELL"
    qty: float
    price: float         # execution price (next-bar open)
    gross: float         # qty * price
    cost: float          # transaction cost + slippage
    cash_delta: float    # signed effect on cash (BUY: -(gross+cost), SELL: +(gross-cost))
    reason: str


def expected_returns_from_forecast(
    buy_set: Sequence[str],
    forecasts: Dict[str, Dict[str, float]],
    closes: Dict[str, float],
    horizon: str = FORECAST_HORIZON,
) -> Dict[str, float]:
    """μ_i = forecast[T+15]/close_i − 1 for each BUY name with a usable forecast."""
    mu: Dict[str, float] = {}
    for t in buy_set:
        fc = forecasts.get(t) or {}
        px = fc.get(horizon)
        c = closes.get(t)
        if px is None or c is None or c <= 0:
            continue
        mu[t] = float(px) / float(c) - 1.0
    return mu


def optimize(
    buy_set: Sequence[str],
    forecasts: Dict[str, Dict[str, float]],
    closes: Dict[str, float],
    returns_hist: Dict[str, Sequence[float]],
    confidence: Dict[str, float],
    config: AllocationConfig,
    *,
    deployable_frac: Optional[float] = None,
    risk_reward: Optional[Dict[str, float]] = None,
) -> OptimizeResult:
    """Max-Sharpe weights (fractions of equity) for the BUY set.

    When ``risk_reward`` is provided, the *total* deployment is conviction-scaled:
    weak BUY sets deploy only a floor and keep cash; only excellent sets exhaust it.
    """
    mu = expected_returns_from_forecast(buy_set, forecasts, closes)
    names = [t for t in buy_set if t in mu and t in returns_hist]
    return optimize_weights(
        names, mu, confidence, returns_hist, config,
        deployable_frac=deployable_frac, risk_reward=risk_reward,
    )


def build_position_views(
    positions: Dict[str, Dict[str, Any]],
    mark_prices: Dict[str, float],
    last_signal: Dict[str, Dict[str, Any]],
) -> Dict[str, PositionView]:
    """Assemble ``PositionView``s (qty, mark price, incumbent risk-reward)."""
    views: Dict[str, PositionView] = {}
    for ticker, pos in positions.items():
        qty = float(pos.get("qty", 0.0))
        if qty <= 0:
            continue
        price = float(mark_prices.get(ticker, pos.get("avg_price", 0.0)) or 0.0)
        sig = last_signal.get(ticker, {}) or {}
        rr = risk_reward_ratio(
            sig.get("buy_price") or pos.get("avg_price"),
            sig.get("sell_price"),
            sig.get("stoploss"),
        )
        views[ticker.upper()] = PositionView(
            ticker=ticker.upper(), qty=qty, price=price,
            risk_reward=rr, action=str(sig.get("action", "HOLD")),
        )
    return views


def build_buy_meta(
    buy_set: Sequence[str],
    closes: Dict[str, float],
    last_signal: Dict[str, Dict[str, Any]],
) -> Dict[str, BuyMeta]:
    """Per-BUY entry price / risk-reward / confidence for the reconciler."""
    meta: Dict[str, BuyMeta] = {}
    for t in buy_set:
        sig = last_signal.get(t, {}) or {}
        price = sig.get("buy_price") or closes.get(t)
        if not price or price <= 0:
            continue
        rr = risk_reward_ratio(price, sig.get("sell_price"), sig.get("stoploss"))
        meta[t.upper()] = BuyMeta(
            ticker=t.upper(), price=float(price),
            risk_reward=rr, confidence=float(sig.get("confidence") or 0.5),
        )
    return meta


def reconcile(
    *,
    cash: float,
    positions: Dict[str, Dict[str, Any]],
    mark_prices: Dict[str, float],
    last_signal: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
    equity: float,
    avoid_set: Sequence[str],
    hold_set: Sequence[str],
    config: AllocationConfig,
    deploy_fraction: Optional[float] = None,
) -> ReconcileResult:
    """Turn target weights into cash-guarded orders (see engine.reconcile_orders).

    ``deploy_fraction`` (the conviction-scaled deployment from the optimizer) sets
    the cash reserve to ``(1 - deploy_fraction) * equity`` so the reconciler keeps
    exactly the cash the sizing intended to hold back.
    """
    views = build_position_views(positions, mark_prices, last_signal)
    # Entry sizing prefers the LLM buy_price, falling back to the mark price.
    buy_meta = build_buy_meta(list(weights), mark_prices, last_signal)
    target_notional = {t: float(w) * float(equity) for t, w in weights.items() if w > 0}
    reserve = None if deploy_fraction is None else max(0.0, (1.0 - float(deploy_fraction)) * float(equity))
    return reconcile_orders(
        cash=float(cash),
        positions=views,
        target_notional=target_notional,
        buy_meta=buy_meta,
        avoid=avoid_set,
        hold=hold_set,
        config=config,
        equity=float(equity),
        min_cash_reserve=reserve,
    )


def fill_exits(
    exits: Sequence[Any],
    positions: Dict[str, Dict[str, Any]],
    *,
    atr_map: Optional[Dict[str, float]] = None,
    cost_bps: float = 40.0,
    slippage_atr_mult: float = 0.10,
) -> List[Fill]:
    """Settle intraday target/stop exits at their trigger price, *this* bar.

    Each exit liquidates the whole held quantity at ``exit.price`` (the target or
    stop level, see :mod:`backtesting.exits` for the fill conventions). Unlike
    ordinary orders these do **not** wait for the next bar's open — a resting
    limit/stop fires the moment its level is touched — so they are filled here and
    applied to the book on the same bar. Transaction cost + ATR slippage still
    apply (slippage always worsens the received price on a SELL).
    """
    atr_map = atr_map or {}
    fills: List[Fill] = []
    for e in exits:
        ticker = e.ticker.upper()
        pos = positions.get(ticker) or positions.get(e.ticker)
        qty = float((pos or {}).get("qty", 0.0))
        price = float(e.price)
        if qty <= 0 or price <= 0:
            continue
        gross = qty * price
        txn = costs_mod.roundtrip_cost_inr(gross, bps=cost_bps)
        slip = costs_mod.atr_slippage_inr(qty, atr_map.get(ticker, float("nan")), price, mult=slippage_atr_mult)
        cost = txn + slip
        fills.append(Fill(
            ticker=ticker, side="SELL", qty=qty, price=price,
            gross=gross, cost=cost, cash_delta=gross - cost,
            reason=f"exit:{e.reason}",
        ))
    return fills


def fill_at_open(
    orders: Sequence[Any],
    open_prices: Dict[str, float],
    *,
    atr_map: Optional[Dict[str, float]] = None,
    cost_bps: float = 40.0,
    slippage_atr_mult: float = 0.10,
) -> List[Fill]:
    """Settle queued orders at the next bar's open, charging cost + slippage.

    Orders whose ticker has no open on the fill day are dropped (not filled) — the
    caller re-queues them if desired. Slippage worsens the effective price (pay up
    on a BUY, receive less on a SELL); the transaction cost is charged on top.
    """
    atr_map = atr_map or {}
    fills: List[Fill] = []
    for o in orders:
        ticker = o.ticker.upper()
        price = open_prices.get(ticker)
        if price is None or price <= 0 or o.qty <= 0:
            continue
        gross = float(o.qty) * float(price)
        txn = costs_mod.roundtrip_cost_inr(gross, bps=cost_bps)
        slip = costs_mod.atr_slippage_inr(o.qty, atr_map.get(ticker, float("nan")), price, mult=slippage_atr_mult)
        cost = txn + slip
        if o.side == "BUY":
            cash_delta = -(gross + cost)
        else:  # SELL
            cash_delta = gross - cost
        fills.append(Fill(
            ticker=ticker, side=o.side, qty=float(o.qty), price=float(price),
            gross=gross, cost=cost, cash_delta=cash_delta, reason=o.reason,
        ))
    return fills
