"""Unit tests for the shared allocation engine (pure math, no I/O)."""
from __future__ import annotations

import numpy as np
import pytest

from allocation.config import AllocationConfig
from allocation.engine import (
    BuyMeta,
    PositionView,
    deployment_fraction,
    max_sharpe_weights,
    optimize_weights,
    per_name_caps,
    reconcile_orders,
    risk_reward_ratio,
    shrunk_covariance,
    tilt_expected_returns,
)


def cfg(**over) -> AllocationConfig:
    base = dict(
        initial_capital=1_000_000.0,
        cash_floor_frac=0.10,
        per_name_cap_frac=0.5,
        vol_target_annual=0.20,
        llm_confidence_tilt=0.25,
        min_ticket_inr=1_000.0,
        reallocation_edge=0.15,
    )
    base.update(over)
    return AllocationConfig(**base)


# --------------------------------------------------------------------------
# Expected-return tilt
# --------------------------------------------------------------------------
def test_tilt_neutral_confidence_is_identity():
    mu = [0.05, -0.03, 0.10]
    out = tilt_expected_returns(mu, [0.5, 0.5, 0.5], 0.25)
    assert np.allclose(out, mu)


def test_tilt_scales_but_never_flips_sign():
    mu = [0.05, -0.05]
    out = tilt_expected_returns(mu, [1.0, 1.0], 0.25)
    assert out[0] == pytest.approx(0.05 * 1.25)
    assert out[1] == pytest.approx(-0.05 * 1.25)
    # Even an extreme downward tilt cannot flip a positive mu negative.
    out2 = tilt_expected_returns([0.05], [0.0], 5.0)
    assert out2[0] >= 0.0


# --------------------------------------------------------------------------
# Covariance + caps
# --------------------------------------------------------------------------
def test_shrunk_cov_is_symmetric_pd():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)) * 0.01
    cov = shrunk_covariance(X)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_per_name_caps_vol_scaled():
    # Flat cap 0.6 so the vol-target term can actually bite for the vol name.
    caps = per_name_caps([0.10, 0.40, 0.0], per_name_cap_frac=0.6, vol_target_annual=0.20)
    assert caps[0] == pytest.approx(0.6)          # 0.20/0.10=2.0 -> flat cap 0.6 wins
    assert caps[1] == pytest.approx(0.5)          # 0.20/0.40=0.5 -> vol cap bites
    assert caps[2] == pytest.approx(0.6)          # zero sigma -> flat cap


# --------------------------------------------------------------------------
# Max-Sharpe optimiser
# --------------------------------------------------------------------------
def test_max_sharpe_matches_closed_form_uncapped_diagonal():
    # Diagonal covariance, generous caps: tangency w ∝ Σ⁻¹μ, renormalised.
    mu = np.array([0.10, 0.05, 0.02])
    cov = np.diag([0.04, 0.04, 0.04])
    caps = np.array([1.0, 1.0, 1.0])
    w = max_sharpe_weights(mu, cov, caps, budget=1.0)
    expected = mu / mu.sum()  # Σ⁻¹μ with equal variances ∝ μ
    assert np.allclose(w, expected, atol=1e-3)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_max_sharpe_respects_caps_and_budget():
    mu = np.array([0.20, 0.05])
    cov = np.diag([0.04, 0.04])
    caps = np.array([0.30, 0.80])
    w = max_sharpe_weights(mu, cov, caps, budget=0.9)
    assert w[0] <= 0.30 + 1e-9
    assert w.sum() == pytest.approx(0.9, abs=1e-6)


def test_max_sharpe_all_negative_mu_is_all_cash():
    w = max_sharpe_weights([-0.1, -0.2], np.diag([0.04, 0.04]), [0.5, 0.5], budget=0.9)
    assert np.allclose(w, 0.0)


# --------------------------------------------------------------------------
# optimize_weights integration: cash floor => partial deployment
# --------------------------------------------------------------------------
def test_optimize_weights_keeps_cash_reserve():
    rng = np.random.default_rng(1)
    hist = {t: list(rng.normal(0, 0.01, 260)) for t in ("A", "B", "C")}
    mu = {"A": 0.06, "B": 0.04, "C": 0.03}
    conf = {"A": 0.8, "B": 0.6, "C": 0.5}
    res = optimize_weights(["A", "B", "C"], mu, conf, hist, cfg())
    total = sum(res.weights.values())
    assert total <= 0.90 + 1e-6          # cash floor 10% retained
    assert all(0 <= w <= 0.5 + 1e-9 for w in res.weights.values())


# --------------------------------------------------------------------------
# Conviction-scaled deployment
# --------------------------------------------------------------------------
def test_deployment_fraction_ramps_with_risk_reward():
    c = cfg(deploy_floor_frac=0.30, rr_min_deploy=1.0, rr_full_deploy=3.0)
    assert deployment_fraction(1.0, c) == pytest.approx(0.30)     # marginal -> floor
    assert deployment_fraction(3.0, c) == pytest.approx(1.00)     # excellent -> exhaust
    assert deployment_fraction(0.5, c) == pytest.approx(0.30)     # below floor clamps
    assert deployment_fraction(5.0, c) == pytest.approx(1.00)     # above full clamps
    mid = deployment_fraction(2.0, c)
    assert 0.30 < mid < 1.0                                        # monotone in between


def test_optimize_weak_rr_keeps_more_cash_than_strong():
    rng = np.random.default_rng(3)
    hist = {t: list(rng.normal(0, 0.01, 260)) for t in ("A", "B")}
    mu = {"A": 0.05, "B": 0.04}
    conf = {"A": 0.7, "B": 0.6}
    c = cfg(deploy_floor_frac=0.30, rr_min_deploy=1.0, rr_full_deploy=3.0)

    weak = optimize_weights(["A", "B"], mu, conf, hist, c, risk_reward={"A": 1.0, "B": 1.0})
    strong = optimize_weights(["A", "B"], mu, conf, hist, c, risk_reward={"A": 4.0, "B": 4.0})

    weak_deployed = sum(weak.weights.values())
    strong_deployed = sum(strong.weights.values())
    assert weak_deployed == pytest.approx(0.30, abs=1e-6)          # partial: keep 70% cash
    assert strong_deployed == pytest.approx(1.00, abs=1e-6)        # exhaust for great deals
    assert strong_deployed > weak_deployed


def test_reconcile_reserve_from_conviction():
    # Weak conviction -> deploy fraction ~0.3 -> reserve ~70% held as cash.
    res = reconcile_orders(
        cash=1_000_000.0, positions={},
        target_notional={"A": 300_000.0}, buy_meta={"A": BuyMeta("A", 100.0, 1.0, 0.6)},
        avoid=[], hold=[], config=cfg(), equity=1_000_000.0,
        min_cash_reserve=700_000.0,
    )
    spent = sum(o.notional for o in res.orders if o.side == "BUY")
    assert spent == pytest.approx(300_000.0, abs=1.0)
    assert res.projected_cash == pytest.approx(700_000.0, abs=1.0)


# --------------------------------------------------------------------------
# risk-reward
# --------------------------------------------------------------------------
def test_risk_reward_basic():
    # entry 100, target 110 (+10%), stop 95 (-5%) -> 2.0
    assert risk_reward_ratio(100, 110, 95) == pytest.approx(2.0)


def test_risk_reward_no_stop_returns_default():
    assert risk_reward_ratio(100, 110, 100) == 1.0
    assert risk_reward_ratio(None, None, None) == 1.0


# --------------------------------------------------------------------------
# Cash-guard reconciler
# --------------------------------------------------------------------------
def test_reconcile_never_spends_more_than_cash():
    res = reconcile_orders(
        cash=100_000.0,
        positions={},
        target_notional={"A": 80_000.0, "B": 80_000.0},   # asks 160k, only 100k cash
        buy_meta={"A": BuyMeta("A", 100.0, 2.0, 0.7), "B": BuyMeta("B", 50.0, 1.5, 0.6)},
        avoid=[],
        hold=[],
        config=cfg(),
    )
    spent = sum(o.notional for o in res.orders if o.side == "BUY")
    assert spent <= 100_000.0 + 1e-6
    assert res.projected_cash >= -1e-9
    # The higher risk-reward name (A) is funded first / fully.
    a_spend = sum(o.notional for o in res.orders if o.side == "BUY" and o.ticker == "A")
    assert a_spend == pytest.approx(80_000.0, abs=1.0)


def test_reconcile_keeps_cash_floor_reserve():
    # All-cash 1M, one BUY wanting 950k, floor 10% -> only 900k deployed.
    res = reconcile_orders(
        cash=1_000_000.0, positions={},
        target_notional={"A": 950_000.0},
        buy_meta={"A": BuyMeta("A", 100.0, 2.0, 0.7)},
        avoid=[], hold=[], config=cfg(cash_floor_frac=0.10), equity=1_000_000.0,
    )
    spent = sum(o.notional for o in res.orders if o.side == "BUY")
    assert spent == pytest.approx(900_000.0, abs=1.0)
    assert res.projected_cash == pytest.approx(100_000.0, abs=1.0)


def test_reconcile_avoid_liquidates_position():
    res = reconcile_orders(
        cash=0.0,
        positions={"X": PositionView("X", qty=100, price=200.0, risk_reward=1.0)},
        target_notional={},
        buy_meta={},
        avoid=["X"],
        hold=[],
        config=cfg(),
    )
    sells = [o for o in res.orders if o.side == "SELL" and o.ticker == "X"]
    assert len(sells) == 1
    assert sells[0].qty == pytest.approx(100)
    assert res.projected_cash == pytest.approx(20_000.0)


def test_reconcile_pulls_capital_from_weaker_holding():
    # Cash is exhausted; incumbent WEAK (rr 1.0) is held; new BUY STRONG (rr 3.0)
    # beats it by well over the 15% edge, so capital is rotated out of WEAK.
    res = reconcile_orders(
        cash=0.0,
        positions={"WEAK": PositionView("WEAK", qty=1000, price=100.0, risk_reward=1.0, action="HOLD")},
        target_notional={"STRONG": 60_000.0},
        buy_meta={"STRONG": BuyMeta("STRONG", price=300.0, risk_reward=3.0, confidence=0.8)},
        avoid=[],
        hold=["WEAK"],
        config=cfg(),
    )
    assert res.reallocations, "expected an explicit reallocation record"
    r = res.reallocations[0]
    assert r["from"] == "WEAK" and r["to"] == "STRONG"
    buy = [o for o in res.orders if o.side == "BUY" and o.ticker == "STRONG"]
    assert buy and buy[0].notional > 0
    assert "WEAK" in buy[0].funded_by
    assert res.projected_cash >= -1e-9


def test_reconcile_does_not_rob_similar_holding():
    # New BUY only marginally better than incumbent (within edge) -> no rotation.
    res = reconcile_orders(
        cash=0.0,
        positions={"HELD": PositionView("HELD", qty=1000, price=100.0, risk_reward=2.0, action="HOLD")},
        target_notional={"NEW": 60_000.0},
        buy_meta={"NEW": BuyMeta("NEW", price=300.0, risk_reward=2.1, confidence=0.6)},
        avoid=[],
        hold=["HELD"],
        config=cfg(),
    )
    assert not res.reallocations
    assert not [o for o in res.orders if o.side == "SELL"]
