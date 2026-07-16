"""Shared quantitative fund-allocation engine.

This package is the **single source of truth** for how discrete LLM BUY/HOLD/AVOID
calls are turned into rupee position sizes. It is imported by *both*:

* the walk-forward backtest (``backtesting.portfolio``), and
* the live production allocation layer (``features.portfolio_allocation``),

so the sizing logic that is validated in the backtest is byte-for-byte the same
logic that advises users in production.

Two responsibilities live here, kept deliberately separate and pure:

1. :mod:`allocation.engine` — Markowitz max-Sharpe sizing of the BUY set
   (μ from the forecast path, tilted by LLM confidence; Σ = Ledoit-Wolf shrunk
   covariance; vol-scaled per-name caps; a cash reserve so capital is only ever
   partially deployed), plus the **cash-guard reconciler** that converts target
   notionals into orders while guaranteeing the book can never go negative and
   pulling capital out of a weaker existing holding only when a new BUY beats it
   on risk-reward by a configured edge.
2. :mod:`allocation.config` — the knobs (``AllocationConfig``), env-loadable for
   production and constructible from ``BacktestSettings`` for the harness.
"""
from __future__ import annotations

from allocation.config import AllocationConfig
from allocation.engine import (
    BuyMeta,
    PositionView,
    Order,
    ReconcileResult,
    max_sharpe_weights,
    optimize_weights,
    per_name_caps,
    reconcile_orders,
    risk_reward_ratio,
    shrunk_covariance,
    tilt_expected_returns,
)

__all__ = [
    "AllocationConfig",
    "BuyMeta",
    "PositionView",
    "Order",
    "ReconcileResult",
    "max_sharpe_weights",
    "optimize_weights",
    "per_name_caps",
    "reconcile_orders",
    "risk_reward_ratio",
    "shrunk_covariance",
    "tilt_expected_returns",
]
