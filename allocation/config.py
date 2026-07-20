"""Configuration for the shared fund-allocation engine.

``AllocationConfig`` collects every tunable the sizing/reconciliation math needs.
It is intentionally decoupled from both ``Settings`` and ``BacktestSettings`` so
the engine has no import dependency on either config object — callers translate
their own config into an ``AllocationConfig`` via the constructors below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


@dataclass(frozen=True)
class AllocationConfig:
    # Book size the weights are scaled against. In the backtest this is ₹1cr; in
    # production it is the user-configured investable capital.
    initial_capital: float = 1_00_00_000.0
    # Fraction of *equity* always retained as cash → capital is only ever
    # partially deployed and the book can never be fully drawn down.
    cash_floor_frac: float = 0.10
    # Hard per-name weight ceiling (fraction of the book).
    per_name_cap_frac: float = 0.20
    # Minimum per-name allocation floor (fraction of the book). Any Markowitz
    # weight below this is dropped to 0 — a position must be at least this size or
    # not taken at all. Freed capital is NOT redistributed; it stays in cash.
    min_weight_frac: float = 0.05
    # Annual vol target used to shrink the per-name cap for volatile names:
    # cap_i = min(per_name_cap_frac, vol_target_annual / sigma_i).
    vol_target_annual: float = 0.20
    # Mild expected-return tilt applied from the LLM confidence in [0,1]:
    # mu_i *= 1 + tilt*(2*conf_i - 1). Never flips the sign of mu.
    llm_confidence_tilt: float = 0.25
    # Orders below this rupee notional are dropped (min ticket / odd-lot guard).
    min_ticket_inr: float = 10_000.0
    # Risk-reward edge a new BUY must beat an incumbent holding by before capital
    # is rotated out of the incumbent to fund the BUY (cash-exhaustion path).
    reallocation_edge: float = 0.15

    # ── Conviction-scaled deployment ─────────────────────────────────────────
    # How MUCH of the book to deploy is NOT fixed — it scales with the BUY set's
    # risk-reward quality. Weak-but-still-BUY opportunities deploy only
    # ``deploy_floor_frac`` (most capital stays in cash); the deployment ramps up
    # with the portfolio's weighted risk-reward and only reaches 100% (exhaust the
    # cash) when the deals are genuinely excellent (weighted RR >= rr_full_deploy).
    deploy_floor_frac: float = 0.30      # min deployment when RR <= rr_min_deploy
    rr_min_deploy: float = 1.0           # weighted RR at/below which we deploy the floor
    rr_full_deploy: float = 3.0          # weighted RR at/above which we deploy everything

    # Annualisation factor for daily covariance (NSE ~252 trading days).
    trading_days_per_year: int = 252
    # Risk-free rate (annual) for the Sharpe objective. Cash earns this.
    risk_free_annual: float = 0.0

    @property
    def deployable_frac(self) -> float:
        return max(0.0, 1.0 - self.cash_floor_frac)

    @classmethod
    def from_env(cls, *, initial_capital: float | None = None) -> "AllocationConfig":
        """Production allocation config from ``ALLOC_*`` env vars."""
        return cls(
            initial_capital=(
                initial_capital
                if initial_capital is not None
                else _float_env("ALLOC_INITIAL_CAPITAL", 1_00_00_000.0)
            ),
            cash_floor_frac=_float_env("ALLOC_CASH_FLOOR_FRAC", 0.10),
            per_name_cap_frac=_float_env("ALLOC_PER_NAME_CAP_FRAC", 0.20),
            min_weight_frac=_float_env("ALLOC_MIN_WEIGHT_FRAC", 0.05),
            vol_target_annual=_float_env("ALLOC_VOL_TARGET_ANNUAL", 0.20),
            llm_confidence_tilt=_float_env("ALLOC_LLM_CONFIDENCE_TILT", 0.25),
            min_ticket_inr=_float_env("ALLOC_MIN_TICKET_INR", 10_000.0),
            reallocation_edge=_float_env("ALLOC_REALLOCATION_EDGE", 0.15),
            deploy_floor_frac=_float_env("ALLOC_DEPLOY_FLOOR_FRAC", 0.30),
            rr_min_deploy=_float_env("ALLOC_RR_MIN_DEPLOY", 1.0),
            rr_full_deploy=_float_env("ALLOC_RR_FULL_DEPLOY", 3.0),
            risk_free_annual=_float_env("ALLOC_RISK_FREE_ANNUAL", 0.0),
        )

    @classmethod
    def from_backtest_settings(cls, settings: Any) -> "AllocationConfig":
        """Translate a ``BacktestSettings`` into an ``AllocationConfig``."""
        return cls(
            initial_capital=float(settings.initial_capital),
            cash_floor_frac=float(settings.cash_floor_frac),
            per_name_cap_frac=float(settings.per_name_cap_frac),
            min_weight_frac=float(getattr(settings, "min_weight_frac", 0.05)),
            vol_target_annual=float(settings.vol_target_annual),
            llm_confidence_tilt=float(settings.llm_confidence_tilt),
            min_ticket_inr=float(settings.min_ticket_inr),
            reallocation_edge=float(settings.reallocation_edge),
            deploy_floor_frac=float(getattr(settings, "deploy_floor_frac", 0.30)),
            rr_min_deploy=float(getattr(settings, "rr_min_deploy", 1.0)),
            rr_full_deploy=float(getattr(settings, "rr_full_deploy", 3.0)),
        )
