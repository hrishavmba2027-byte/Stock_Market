"""Tests for the production fund-allocation layer (engine-driven, no network)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from allocation.config import AllocationConfig
from features.portfolio_allocation import (
    Holdings,
    compute_allocation,
    write_bookkeeping,
)


def cfg(**over) -> AllocationConfig:
    base = dict(initial_capital=1_000_000.0, cash_floor_frac=0.10, per_name_cap_frac=0.5,
                vol_target_annual=0.20, llm_confidence_tilt=0.25, min_ticket_inr=1_000.0,
                reallocation_edge=0.15)
    base.update(over)
    return AllocationConfig(**base)


def _hist(seed=0, n=260):
    rng = np.random.default_rng(seed)
    return list(rng.normal(0, 0.01, n))


def test_all_cash_strong_deals_deploy_heavily():
    # A: RR=(115-100)/(100-95)=3.0, B: RR=(55-50)/(50-48)=2.5 -> excellent deals,
    # so conviction deploys most/all of the book (partial-only for weak deals).
    suggestions = {
        "A": {"action": "BUY", "buy_price": 100, "sell_price": 115, "stoploss": 95, "confidence": 0.8},
        "B": {"action": "BUY", "buy_price": 50, "sell_price": 55, "stoploss": 48, "confidence": 0.6},
        "C": {"action": "AVOID"},
    }
    forecasts = {
        "A": {"close": 100, "forecast": {"T+15": 112}},
        "B": {"close": 50, "forecast": {"T+15": 54}},
        "C": {"close": 30, "forecast": {"T+15": 28}},
    }
    returns_hist = {"A": _hist(1), "B": _hist(2), "C": _hist(3)}
    plan = compute_allocation(
        suggestions=suggestions, forecasts=forecasts, returns_hist=returns_hist,
        holdings=Holdings.all_cash(1_000_000), prices={}, config=cfg(per_name_cap_frac=0.6),
    )
    deployed = plan.equity - plan.projected_cash
    assert deployed >= 0.85 * plan.equity       # strong deals -> heavy deployment
    assert plan.projected_cash >= 0              # never negative
    buys = [a for a in plan.advice if a.recommended_order.startswith("BUY")]
    assert {a.ticker for a in buys} == {"A", "B"}
    assert plan.opt.deploy_fraction > 0.85


def test_weak_deals_keep_more_cash():
    # Reward barely above risk -> low conviction -> deploy the floor, keep cash.
    suggestions = {
        "A": {"action": "BUY", "buy_price": 100, "sell_price": 101, "stoploss": 99, "confidence": 0.55},
        "B": {"action": "BUY", "buy_price": 50, "sell_price": 50.5, "stoploss": 49.5, "confidence": 0.5},
    }
    forecasts = {"A": {"close": 100, "forecast": {"T+15": 101}}, "B": {"close": 50, "forecast": {"T+15": 50.4}}}
    plan = compute_allocation(
        suggestions=suggestions, forecasts=forecasts, returns_hist={"A": _hist(1), "B": _hist(2)},
        holdings=Holdings.all_cash(1_000_000), prices={},
        config=cfg(deploy_floor_frac=0.30, rr_min_deploy=1.0, rr_full_deploy=3.0),
    )
    deployed = plan.equity - plan.projected_cash
    assert deployed <= 0.45 * plan.equity        # weak deals -> keep most cash at hand
    assert plan.opt.deploy_fraction <= 0.4


def test_hold_is_frozen_and_avoid_exits():
    suggestions = {
        "H": {"action": "HOLD"},
        "X": {"action": "AVOID"},
    }
    forecasts = {"H": {"close": 100, "forecast": {"T+15": 100}}, "X": {"close": 100, "forecast": {"T+15": 100}}}
    holdings = Holdings(capital=1_000_000, cash=200_000,
                        positions={"H": {"qty": 1000, "avg_price": 100}, "X": {"qty": 500, "avg_price": 100}})
    plan = compute_allocation(
        suggestions=suggestions, forecasts=forecasts, returns_hist={}, holdings=holdings,
        prices={"H": 100, "X": 100}, config=cfg(),
    )
    adv = {a.ticker: a for a in plan.advice}
    assert adv["H"].recommended_order == "HOLD"
    assert adv["X"].recommended_order.startswith("EXIT")


def test_cash_exhaustion_rotates_from_weaker_holding():
    # No free cash; weak incumbent held; strong BUY beats it on risk-reward.
    suggestions = {
        "WEAK": {"action": "HOLD", "buy_price": 100, "sell_price": 102, "stoploss": 98},   # rr ~1
        "STRONG": {"action": "BUY", "buy_price": 300, "sell_price": 360, "stoploss": 290, "confidence": 0.8},  # rr ~6
    }
    forecasts = {
        "WEAK": {"close": 100, "forecast": {"T+15": 101}},
        "STRONG": {"close": 300, "forecast": {"T+15": 345}},
    }
    holdings = Holdings(capital=100_000, cash=0.0, positions={"WEAK": {"qty": 1000, "avg_price": 100}})
    plan = compute_allocation(
        suggestions=suggestions, forecasts=forecasts, returns_hist={"STRONG": _hist(5)},
        holdings=holdings, prices={"WEAK": 100, "STRONG": 300}, config=cfg(),
    )
    assert plan.reallocations, "expected a capital rotation record"
    assert plan.reallocations[0]["from"] == "WEAK"
    assert plan.reallocations[0]["to"] == "STRONG"
    strong = next(a for a in plan.advice if a.ticker == "STRONG")
    assert strong.recommended_order.startswith("BUY")
    assert "WEAK" in strong.funded_by
    assert plan.projected_cash >= -1e-6


def test_bookkeeping_workbook_has_sheets(tmp_path: Path):
    suggestions = {"A": {"action": "BUY", "buy_price": 100, "sell_price": 115, "stoploss": 95, "confidence": 0.7}}
    forecasts = {"A": {"close": 100, "forecast": {"T+15": 110}}}
    plan = compute_allocation(
        suggestions=suggestions, forecasts=forecasts, returns_hist={"A": _hist(7)},
        holdings=Holdings.all_cash(1_000_000), prices={}, config=cfg(),
    )
    out = write_bookkeeping(plan, tmp_path / "prod.xlsx", as_of="2026-07-14")
    assert out.exists()
    xl = pd.ExcelFile(out)
    for sheet in ("LLM_Suggestions", "Fund_Allocation", "Reallocations", "Summary"):
        assert sheet in xl.sheet_names
    alloc = pd.read_excel(out, sheet_name="Fund_Allocation")
    assert (alloc["ticker"] == "A").any()
