"""Tests for the deterministic entry gates and the hold-and-trail exit scan."""
from __future__ import annotations

import math

import pytest

from backtesting.exits import scan_exits_and_trail
from features.entry_gates import (
    GateConfig,
    apply_entry_cap,
    apply_entry_gates,
    compute_entry_features,
    effective_rr,
    forecast_change,
)


def _closes(start: float, drift: float, n: int = 40):
    return [start * (1 + drift) ** i for i in range(n)]


def cfg(**over) -> GateConfig:
    base = dict(momentum_lookback_days=15, rsi_min=45.0, rsi_max=70.0,
                volume_spike_ratio=1.5, stop_atr_mult=2.0, min_eff_rr=1.5,
                roundtrip_cost_pct=0.009, max_chase_pct=0.03)
    base.update(over)
    return GateConfig(**base)


def _buy(close=100.0, stop=97.0, target=110.0):
    return {"action": "BUY", "buy_price": close, "sell_price": target,
            "stoploss": stop, "confidence": 0.7, "rationale": "test"}


# ── features ────────────────────────────────────────────────────────────────

def test_momentum_positive_on_uptrend():
    f = compute_entry_features(_closes(100, 0.01), lookback=15)
    assert f["momentum"] > 0
    assert f["rsi14"] > 50


def test_volume_ratio_detects_spike():
    vols = [1000.0] * 39 + [2500.0]
    f = compute_entry_features(_closes(100, 0.001), volumes=vols, lookback=15)
    assert f["volume_ratio"] == pytest.approx(2.5, rel=0.01)


def test_forecast_change_uses_configured_horizon_with_fallback():
    path = {f"T+{h}": 100 + h for h in range(1, 16)}
    assert forecast_change(path, 100.0, 15) == pytest.approx(0.15)
    assert forecast_change(path, 100.0, 30) == pytest.approx(0.15)  # falls back to T+15
    assert forecast_change({}, 100.0, 15) is None


# ── gates ───────────────────────────────────────────────────────────────────

def test_gate_blocks_negative_momentum():
    closes = _closes(100, -0.01)
    f = compute_entry_features(closes, lookback=15)
    path = {"T+15": closes[-1] * 1.05}
    out = apply_entry_gates(_buy(close=closes[-1]), f, path, closes[-1], cfg())
    assert out["action"] == "HOLD"
    assert any("momentum" in r for r in out["gate_reasons"])


def test_gate_blocks_negative_forecast_even_with_high_confidence():
    closes = _closes(100, 0.01)          # rising stock
    f = compute_entry_features(closes, lookback=15)
    path = {"T+15": closes[-1] * 0.97}   # bearish forecast
    sug = _buy(close=closes[-1]); sug["confidence"] = 0.95
    out = apply_entry_gates(sug, f, path, closes[-1], cfg())
    assert out["action"] == "HOLD"       # override is off by default (never fires)
    assert any("forecast" in r for r in out["gate_reasons"])


def test_gate_widens_tight_stop_to_atr_floor():
    closes = _closes(100, 0.005)
    last = closes[-1]
    f = compute_entry_features(closes, highs=[c * 1.02 for c in closes],
                               lows=[c * 0.98 for c in closes], lookback=15)
    path = {"T+15": last * 1.30}
    sug = _buy(close=last, stop=last * 0.995, target=last * 1.30)  # stop 0.5% away
    out = apply_entry_gates(sug, f, path, last, cfg())
    floor = last * (1 - 2.0 * f["atr_pct"])
    assert out["stoploss"] == pytest.approx(round(floor, 2))
    assert out.get("gate_adjustments")


def test_gate_rejects_poor_cost_adjusted_rr():
    closes = _closes(100, 0.005)
    last = closes[-1]
    f = compute_entry_features(closes, lookback=15)
    path = {"T+15": last * 1.01}
    # target 1% above, stop 3% below → eff RR far under 1.5
    out = apply_entry_gates(_buy(close=last, stop=last * 0.97, target=last * 1.01),
                            f, path, last, cfg())
    assert out["action"] == "HOLD"
    assert any("RR" in r for r in out["gate_reasons"])


def test_gate_passes_clean_setup_untouched():
    # zigzag uptrend (+1.2% / −0.6% alternating) → RSI in the healthy band
    closes = [100.0]
    for i in range(39):
        closes.append(closes[-1] * (1.012 if i % 2 == 0 else 0.994))
    last = closes[-1]
    f = compute_entry_features(closes, highs=[c * 1.008 for c in closes],
                               lows=[c * 0.992 for c in closes],
                               volumes=[1000.0] * 40, lookback=15)
    assert 45 <= f["rsi14"] <= 70
    path = {"T+15": last * 1.12}
    sug = _buy(close=last, stop=last * (1 - 2.5 * f["atr_pct"]), target=last * 1.12)
    out = apply_entry_gates(sug, f, path, last, cfg())
    assert out["action"] == "BUY"
    assert "gate_reasons" not in out


def test_non_buy_passes_through():
    out = apply_entry_gates({"action": "AVOID"}, {}, None, None, cfg())
    assert out["action"] == "AVOID"


def test_default_min_eff_rr_is_one():
    # 2026-07-20: relaxed from 1.5 (blocked 50/76 BUYs incl. all of 2021) to 1.0
    # (rejects only inverted geometry) — hold-and-trail no longer caps upside
    # at the stated target, so demanding 1.5x of it was inconsistent.
    assert GateConfig().min_eff_rr == pytest.approx(1.0)
    assert GateConfig.from_env().min_eff_rr == pytest.approx(1.0)


# ── per-cycle entry cap (time-diversification) ──────────────────────────────

def _sig(action="BUY", buy=100.0, target=110.0, stop=95.0, confidence=0.7):
    return {"action": action, "buy_price": buy, "last_close": buy,
            "sell_price": target, "stoploss": stop, "confidence": confidence}


def test_effective_rr_matches_gate_arithmetic():
    # target 10% above, stop 5% below, 0.9% cost: reward=0.091, risk=0.059
    rr = effective_rr(100.0, 110.0, 95.0, cost_pct=0.009)
    assert rr == pytest.approx((0.10 - 0.009) / (0.05 + 0.009))


def test_entry_cap_keeps_top_n_by_rr_defers_rest():
    suggestions = {
        "A": _sig(target=130.0, stop=97.0),   # best RR
        "B": _sig(target=120.0, stop=97.0),
        "C": _sig(target=112.0, stop=97.0),
        "D": _sig(target=108.0, stop=97.0),   # worst RR
        "E": _sig(action="HOLD"),             # not a fresh BUY — never touched
    }
    deferred = apply_entry_cap(suggestions, held=[], max_new=2)
    assert set(deferred) == {"C", "D"}
    assert suggestions["A"]["action"] == "BUY"
    assert suggestions["B"]["action"] == "BUY"
    assert suggestions["C"]["action"] == "HOLD"
    assert suggestions["D"]["action"] == "HOLD"
    assert any("entry cap" in r for r in suggestions["C"]["gate_reasons"])
    assert suggestions["E"]["action"] == "HOLD"  # untouched, was never a BUY


def test_entry_cap_ignores_already_held_names():
    suggestions = {
        "A": _sig(target=130.0), "B": _sig(target=120.0), "C": _sig(target=110.0),
    }
    # "A" is already held — a BUY there (e.g. a re-underwrite) doesn't count
    # against the cap on NEW names, so both B and C (2 new names, cap 1 would
    # normally defer one) ... here cap=1 with A excluded leaves 2 fresh names.
    deferred = apply_entry_cap(suggestions, held=["A"], max_new=1)
    assert suggestions["A"]["action"] == "BUY"       # held name never capped
    assert len(deferred) == 1
    assert deferred[0] in ("B", "C")


def test_entry_cap_disabled_at_zero():
    suggestions = {"A": _sig(), "B": _sig(), "C": _sig(), "D": _sig()}
    deferred = apply_entry_cap(suggestions, held=[], max_new=0)
    assert deferred == []
    assert all(s["action"] == "BUY" for s in suggestions.values())


def test_entry_cap_noop_when_under_limit():
    suggestions = {"A": _sig(), "B": _sig()}
    deferred = apply_entry_cap(suggestions, held=[], max_new=3)
    assert deferred == []


# ── hold-and-trail exits ────────────────────────────────────────────────────

def _pos(qty=10, avg=100.0, entry_bar="2024-01-01"):
    return {"T": {"qty": qty, "avg_price": avg, "entry_bar": entry_bar}}


def test_target_touch_arms_trail_instead_of_selling():
    sig = {"T": {"sell_price": 110.0, "stoploss": 95.0}}
    ohlc = {"T": {"open": 108.0, "high": 112.0, "low": 107.0, "close": 111.0}}
    exits, updates = scan_exits_and_trail(_pos(), sig, ohlc, trail_on_target=True,
                                          trail_atr_mult=2.0, atr_map={"T": 2.0})
    assert exits == []
    assert updates["T"]["trail_armed"] is True
    # max(prev stop 95, close-2*ATR=107, entry 100) = 107
    assert updates["T"]["stoploss"] == pytest.approx(107.0)


def test_armed_trail_ratchets_up_only():
    sig = {"T": {"sell_price": 110.0, "stoploss": 107.0, "trail_armed": True}}
    ohlc = {"T": {"open": 112.0, "high": 115.0, "low": 111.0, "close": 114.0}}
    exits, updates = scan_exits_and_trail(_pos(), sig, ohlc, trail_on_target=True,
                                          trail_atr_mult=2.0, atr_map={"T": 2.0})
    assert exits == []
    assert updates["T"]["stoploss"] == pytest.approx(110.0)   # 114 − 4

    # a lower close must NOT lower the stop
    ohlc2 = {"T": {"open": 109.0, "high": 110.5, "low": 108.0, "close": 108.5}}
    sig2 = {"T": {"sell_price": 110.0, "stoploss": 110.0, "trail_armed": True}}
    exits2, updates2 = scan_exits_and_trail(_pos(), sig2, ohlc2, trail_on_target=True,
                                            trail_atr_mult=2.0, atr_map={"T": 2.0})
    assert "T" not in updates2 or updates2["T"]["stoploss"] >= 110.0


def test_stop_hit_still_sells_and_beats_target_ambiguity():
    sig = {"T": {"sell_price": 110.0, "stoploss": 95.0}}
    ohlc = {"T": {"open": 100.0, "high": 111.0, "low": 94.0, "close": 96.0}}
    exits, updates = scan_exits_and_trail(_pos(), sig, ohlc, trail_on_target=True,
                                          atr_map={"T": 2.0})
    assert len(exits) == 1 and exits[0].reason == "stop"
    assert updates == {}


def test_max_age_exit_fires():
    sig = {"T": {"sell_price": None, "stoploss": 90.0}}
    ohlc = {"T": {"open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5}}
    exits, _ = scan_exits_and_trail(_pos(entry_bar="2024-01-01"), sig, ohlc,
                                    trail_on_target=True, atr_map={"T": 2.0},
                                    max_holding_days=30, bar="2024-02-15")
    assert len(exits) == 1 and exits[0].reason == "max_age"
    assert exits[0].price == pytest.approx(101.5)


def test_legacy_target_sell_when_trailing_disabled():
    sig = {"T": {"sell_price": 110.0, "stoploss": 95.0}}
    ohlc = {"T": {"open": 108.0, "high": 112.0, "low": 107.0, "close": 111.0}}
    exits, updates = scan_exits_and_trail(_pos(), sig, ohlc, trail_on_target=False)
    assert len(exits) == 1 and exits[0].reason == "target"
    assert updates == {}


# ── position-review profit-cushion guard (ULTRACEMCO whipsaw fix) ───────────
from features.trade_suggestions import _translate_review


def _position(entry=6047.12, current=6099.13, stop=5683.07, target=6768.05,
              arm=6168.0, armed=False):
    return {"entry_price_inr": entry, "current_price_inr": current,
            "current_stoploss": stop, "current_target": target,
            "min_price_to_trail_inr": arm, "trail_armed": armed}


def test_review_clamps_premature_breakeven_raise():
    # Held ~1 day, +0.86% — below the arm level. Review tries to raise stop to
    # entry (break-even). Guard must keep the original wide stop.
    review = {"decision": "HOLD_AND_RAISE", "new_target": 6768.05,
              "new_stoploss": 6047.12, "confidence": 0.85, "rationale": "x"}
    out = _translate_review(review, _position(current=6099.13, arm=6168.0))
    assert out["action"] == "HOLD"
    assert out["stoploss"] == pytest.approx(5683.07)   # NOT raised to break-even
    assert "not yet" in out["rationale"]


def test_review_allows_breakeven_raise_when_cushion_cleared():
    review = {"decision": "HOLD_AND_RAISE", "new_target": 6900.0,
              "new_stoploss": 6047.12, "confidence": 0.85, "rationale": "x"}
    out = _translate_review(review, _position(current=6200.0, arm=6168.0))
    assert out["stoploss"] == pytest.approx(6047.12)   # cushion cleared → allowed


def test_review_allows_breakeven_when_trail_already_armed():
    review = {"decision": "HOLD_AND_RAISE", "new_stoploss": 6100.0,
              "confidence": 0.8, "rationale": "x"}
    out = _translate_review(review, _position(current=6050.0, arm=9999.0, armed=True))
    assert out["stoploss"] == pytest.approx(6100.0)


def test_review_allows_below_entry_tightening_without_cushion():
    # Raising 5683 → 5900 (still below entry) is pure risk reduction, no whipsaw.
    review = {"decision": "HOLD_AND_RAISE", "new_stoploss": 5900.0,
              "confidence": 0.7, "rationale": "x"}
    out = _translate_review(review, _position(current=6099.13, arm=6168.0))
    assert out["stoploss"] == pytest.approx(5900.0)


def test_review_stop_never_ratchets_down():
    review = {"decision": "HOLD", "new_stoploss": 5000.0, "confidence": 0.6, "rationale": "x"}
    out = _translate_review(review, _position(stop=5683.07))
    assert out["stoploss"] == pytest.approx(5683.07)   # floored at previous stop


def test_review_exit_maps_to_avoid():
    review = {"decision": "EXIT", "new_stoploss": 6047.12, "confidence": 0.5, "rationale": "fc down"}
    out = _translate_review(review, _position())
    assert out["action"] == "AVOID" and out["review_decision"] == "EXIT"


def test_review_falls_back_to_default_cushion_without_arm_level():
    # No min_price_to_trail_inr supplied → 2% over entry default; +0.9% must clamp.
    pos = _position(current=6099.13, arm=None)
    review = {"decision": "HOLD_AND_RAISE", "new_stoploss": 6047.12, "confidence": 0.8, "rationale": "x"}
    out = _translate_review(review, pos)
    assert out["stoploss"] == pytest.approx(5683.07)


# ── winner-protection guard: EXIT on a profitable position → tight trail ────

def test_exit_on_losing_position_still_maps_to_avoid():
    # Below the arm level, not armed → not "in profit" → EXIT executes as normal.
    review = {"decision": "EXIT", "new_stoploss": 6047.12, "confidence": 0.5, "rationale": "fc down"}
    out = _translate_review(review, _position(current=6099.13, arm=6168.0, armed=False))
    assert out["action"] == "AVOID"
    assert out["review_decision"] == "EXIT"


def test_exit_on_profitable_position_converts_to_tight_trail_not_avoid():
    # Trail already armed (target was hit) — a profitable winner. EXIT must NOT
    # become a market sell; it must ratchet a tight stop instead.
    pos = _position(entry=100.0, current=130.0, stop=118.0, target=140.0, armed=True)
    pos["atr_inr"] = 3.0
    review = {"decision": "EXIT", "new_stoploss": 100.0, "confidence": 0.6,
              "rationale": "momentum stalling"}
    out = _translate_review(review, pos)
    assert out["action"] == "HOLD"
    assert out["review_decision"] == "EXIT_TO_TRAIL"
    # tightened ~1 ATR below current (130 - 3 = 127), never below prev stop (118)
    assert out["stoploss"] == pytest.approx(127.0)
    assert out["stoploss"] < pos["current_price_inr"]
    assert out["stoploss"] >= pos["current_stoploss"]
    assert out["trail_armed"] is True


def test_exit_on_profitable_position_never_drops_below_previous_stop():
    # ATR wide enough that current-ATR would fall below the existing stop —
    # the previous (already-protective) stop must win.
    pos = _position(entry=100.0, current=130.0, stop=125.0, target=140.0, armed=True)
    pos["atr_inr"] = 20.0   # 130 - 20 = 110, below prev stop 125
    review = {"decision": "EXIT", "confidence": 0.6, "rationale": "x"}
    out = _translate_review(review, pos)
    assert out["action"] == "HOLD"
    assert out["stoploss"] == pytest.approx(125.0)


def test_exit_on_position_past_cushion_without_armed_flag_also_converts():
    # Not explicitly "armed", but price has cleared the profit-cushion arm
    # level — still counts as meaningfully in profit.
    pos = _position(entry=100.0, current=120.0, stop=101.0, target=130.0,
                    arm=115.0, armed=False)
    pos["atr_inr"] = 2.0
    review = {"decision": "EXIT", "confidence": 0.55, "rationale": "x"}
    out = _translate_review(review, pos)
    assert out["action"] == "HOLD"
    assert out["review_decision"] == "EXIT_TO_TRAIL"
