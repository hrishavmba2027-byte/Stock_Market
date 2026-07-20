"""Deterministic entry gates for BUY suggestions — shared by backtest and production.

The 2020–2026 walk-forward trade-quality analysis (backtest_trade_quality_report.md)
showed the LLM's raw BUYs lose primarily through four mechanisms: buying falling
stocks against the forecast, buying capitulation-volume days, stops set inside one
daily range, and targets that do not clear transaction costs. These gates enforce
the confirmed rules *deterministically* — the LLM prompt asks for them, this module
guarantees them. A BUY that fails a hard gate is downgraded to HOLD with the reason
recorded in the rationale; a too-tight stop is first widened to the ATR floor and
only rejected if the cost-adjusted reward/risk no longer clears the bar.

Numbers behind the defaults (668 closed round trips):

* momentum+forecast agreement (both > 0 over 15 days): 35.6% win rate vs 28.7%
  baseline; the 15-day lookback maximized both win rate and per-trade P&L across
  the {5,10,15,20,25,30}-day grid, so 15 is the default and configurable.
* volume >= 1.5x 20-day average at entry: 18.9% win rate (17.3% when momentum was
  also negative) → hard reject.
* RSI(14) outside [45, 70]: the sub-30 "oversold" zone was the worst bucket
  (23.0% win rate, −₹16,924/trade).
* stop closer than 2x the daily range: 186 same-bar stop-outs, −₹4.0M.
* cost-adjusted reward/risk < 1.5 (at 0.9% round-trip cost): the only
  positive-P&L slice found required >= 1.5.

Everything here is pure math over plain floats/dicts — no network, no sheets —
so both ``backtesting.signals`` and ``features.trade_suggestions`` call it with
whatever history they have. A gate whose inputs are unavailable (e.g. no volume
history in a production payload) is skipped rather than guessed.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


@dataclass(frozen=True)
class GateConfig:
    # Lookback (trading days) for the historical-momentum leg AND the forecast
    # horizon (T+N) for the forecast leg of the agreement gate. 15 maximized
    # win rate and per-trade P&L on the 2020–2026 backtest grid.
    momentum_lookback_days: int = 15
    # RSI(14) band a BUY entry must sit in.
    rsi_min: float = 45.0
    rsi_max: float = 70.0
    # Entry-day volume above this multiple of its 20-day average → reject.
    volume_spike_ratio: float = 1.5
    # Stop must sit at least this many ATR(14) below entry; tighter stops are
    # widened to this floor before the reward/risk check.
    stop_atr_mult: float = 2.0
    # Cost-adjusted reward/risk floor: (tgt% − cost) / (stop% + cost) >= this.
    # 1.0 rejects only inverted geometry (risk outweighing stated reward after
    # costs). The original 1.5 was calibrated for the sell-at-target world; with
    # hold-and-trail the upside is not capped at the stated target, and at 1.5
    # the gate blocked 50/76 BUY intents including every 2021 entry.
    min_eff_rr: float = 1.0
    # Assumed round-trip transaction cost as a fraction of entry.
    roundtrip_cost_pct: float = 0.009
    # Entry (buy_price) farther than this from the latest close → reject (chasing).
    max_chase_pct: float = 0.03
    # Judicious fallback for a negative forecast leg (user rule): allow the BUY
    # anyway only when BOTH the historical momentum is strongly positive AND the
    # computed confidence is very high. Disabled (0.0 threshold ⇒ never) by
    # default because the backtest showed confidence/sentiment could not rescue
    # a negative forecast (22.5% win rate vs 31.8% for low-confidence).
    negative_forecast_override_min_confidence: float = 1.01  # > 1 ⇒ never fires
    negative_forecast_override_min_momentum: float = 0.06

    @classmethod
    def from_env(cls) -> "GateConfig":
        return cls(
            momentum_lookback_days=_int_env("GATE_MOMENTUM_LOOKBACK_DAYS", 15),
            rsi_min=_float_env("GATE_RSI_MIN", 45.0),
            rsi_max=_float_env("GATE_RSI_MAX", 70.0),
            volume_spike_ratio=_float_env("GATE_VOLUME_SPIKE_RATIO", 1.5),
            stop_atr_mult=_float_env("GATE_STOP_ATR_MULT", 2.0),
            min_eff_rr=_float_env("GATE_MIN_EFF_RR", 1.0),
            roundtrip_cost_pct=_float_env("GATE_ROUNDTRIP_COST_PCT", 0.009),
            max_chase_pct=_float_env("GATE_MAX_CHASE_PCT", 0.03),
            negative_forecast_override_min_confidence=_float_env(
                "GATE_NEG_FC_OVERRIDE_MIN_CONFIDENCE", 1.01),
            negative_forecast_override_min_momentum=_float_env(
                "GATE_NEG_FC_OVERRIDE_MIN_MOMENTUM", 0.06),
        )

    @classmethod
    def from_backtest_settings(cls, settings: Any) -> "GateConfig":
        return cls(
            momentum_lookback_days=int(getattr(settings, "momentum_lookback_days", 15)),
            rsi_min=float(getattr(settings, "gate_rsi_min", 45.0)),
            rsi_max=float(getattr(settings, "gate_rsi_max", 70.0)),
            volume_spike_ratio=float(getattr(settings, "gate_volume_spike_ratio", 1.5)),
            stop_atr_mult=float(getattr(settings, "gate_stop_atr_mult", 2.0)),
            min_eff_rr=float(getattr(settings, "gate_min_eff_rr", 1.0)),
            roundtrip_cost_pct=float(getattr(settings, "gate_roundtrip_cost_pct", 0.009)),
            max_chase_pct=float(getattr(settings, "gate_max_chase_pct", 0.03)),
            negative_forecast_override_min_confidence=float(
                getattr(settings, "gate_neg_fc_override_min_confidence", 1.01)),
            negative_forecast_override_min_momentum=float(
                getattr(settings, "gate_neg_fc_override_min_momentum", 0.06)),
        )


# ---------------------------------------------------------------------------
# Feature computation from plain price/volume history
# ---------------------------------------------------------------------------

def _clean(values: Optional[Sequence[Any]]) -> List[float]:
    out: List[float] = []
    for v in values or []:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > 0:
            out.append(f)
    return out


def compute_entry_features(
    closes: Sequence[Any],
    *,
    highs: Optional[Sequence[Any]] = None,
    lows: Optional[Sequence[Any]] = None,
    volumes: Optional[Sequence[Any]] = None,
    lookback: int = 15,
) -> Dict[str, Optional[float]]:
    """Momentum / RSI / ATR% / volume-ratio at the last close of ``closes``.

    ``closes`` must be ordered oldest → newest and end at the decision close.
    Any feature whose history is too short comes back ``None`` (its gate is
    then skipped, never guessed). When highs/lows are absent, ATR%% falls back
    to a close-to-close true-range approximation so the stop-distance floor
    still works from a closes-only payload (production sheets path).
    """
    c = _clean(closes)
    feats: Dict[str, Optional[float]] = {
        "momentum": None, "rsi14": None, "atr_pct": None, "volume_ratio": None,
    }
    n = len(c)
    if n >= lookback + 1:
        feats["momentum"] = c[-1] / c[-1 - lookback] - 1.0
    if n >= 15:
        gains = losses = 0.0
        for i in range(n - 14, n):
            d = c[i] - c[i - 1]
            if d >= 0:
                gains += d
            else:
                losses -= d
        if losses <= 1e-12:
            feats["rsi14"] = 100.0
        else:
            rs = (gains / 14.0) / (losses / 14.0)
            feats["rsi14"] = 100.0 - 100.0 / (1.0 + rs)
    h, l = _clean(highs), _clean(lows)
    if len(h) == n and len(l) == n and n >= 15:
        trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
               for i in range(n - 14, n)]
        feats["atr_pct"] = (sum(trs) / len(trs)) / c[-1]
    elif n >= 15:
        # Closes-only fallback: mean absolute daily change ≈ a conservative
        # stand-in for true range (understates it, so the stop floor it drives
        # is never looser than the real-ATR floor would be… tighter is the
        # dangerous direction, so scale up by the empirical TR/|Δclose| ≈ 1.6).
        deltas = [abs(c[i] - c[i - 1]) for i in range(n - 14, n)]
        feats["atr_pct"] = 1.6 * (sum(deltas) / len(deltas)) / c[-1]
    v = _clean(volumes)
    if len(v) >= 21:
        avg20 = sum(v[-21:-1]) / 20.0
        if avg20 > 0:
            feats["volume_ratio"] = v[-1] / avg20
    return feats


def forecast_change(
    forecast_path: Optional[Dict[str, Any]],
    close: Optional[float],
    horizon_days: int,
) -> Optional[float]:
    """Forecast leg of the agreement gate: forecast[T+N]/close − 1.

    Falls back to the nearest available horizon at/below N (the path may be
    shorter than the configured lookback on legacy 15-day forecasts).
    """
    if not forecast_path or not close or close <= 0:
        return None
    for h in range(int(horizon_days), 0, -1):
        px = forecast_path.get(f"T+{h}")
        try:
            f = float(px)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > 0:
            return f / float(close) - 1.0
    return None


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def apply_entry_gates(
    suggestion: Dict[str, Any],
    features: Dict[str, Optional[float]],
    forecast_path: Optional[Dict[str, Any]],
    close: Optional[float],
    config: GateConfig,
) -> Dict[str, Any]:
    """Enforce the entry gates on a raw LLM suggestion.

    Returns the (possibly modified) suggestion: BUYs that fail a hard gate are
    downgraded to HOLD with ``gate_reasons`` recording why; a too-tight stop is
    widened to the ATR floor (recorded in ``gate_adjustments``) before the
    reward/risk check. Non-BUY actions pass through untouched.
    """
    if str(suggestion.get("action", "")).upper() != "BUY":
        return suggestion

    reasons: List[str] = []
    adjustments: List[str] = []
    out = dict(suggestion)

    mom = features.get("momentum")
    fc = forecast_change(forecast_path, close, config.momentum_lookback_days)

    # Gate 1 — trend agreement. Both legs must be positive. The judicious
    # override for a negative forecast leg fires only when explicitly enabled
    # via config AND momentum is strongly positive AND confidence clears a very
    # high bar (the backtest showed confidence cannot normally rescue this).
    if mom is not None and mom <= 0:
        reasons.append(f"momentum {mom:+.2%} <= 0 over {config.momentum_lookback_days}d")
    if fc is not None and fc <= 0:
        conf = _num(suggestion.get("confidence")) or 0.0
        override = (
            conf >= config.negative_forecast_override_min_confidence
            and mom is not None
            and mom >= config.negative_forecast_override_min_momentum
        )
        if override:
            adjustments.append(f"negative forecast {fc:+.2%} overridden "
                               f"(conf {conf:.2f}, momentum {mom:+.2%})")
        else:
            reasons.append(f"forecast T+{config.momentum_lookback_days} {fc:+.2%} <= 0")

    # Gate 2 — healthy entry: RSI band and no volume spike.
    rsi = features.get("rsi14")
    if rsi is not None and not (config.rsi_min <= rsi <= config.rsi_max):
        reasons.append(f"RSI {rsi:.0f} outside [{config.rsi_min:.0f}, {config.rsi_max:.0f}]")
    vr = features.get("volume_ratio")
    if vr is not None and vr >= config.volume_spike_ratio:
        reasons.append(f"volume {vr:.1f}x 20d average (>= {config.volume_spike_ratio:.1f}x)")

    # Gate 5 — no chasing.
    buy = _num(suggestion.get("buy_price"))
    if buy is not None and close and close > 0:
        chase = abs(buy / float(close) - 1.0)
        if chase > config.max_chase_pct:
            reasons.append(f"entry {chase:.1%} from close (> {config.max_chase_pct:.0%})")

    # Gate 3 — survivable stop: widen to the ATR floor rather than reject.
    stop = _num(suggestion.get("stoploss"))
    atr_pct = features.get("atr_pct")
    entry_ref = buy if buy is not None else (float(close) if close else None)
    if stop is not None and entry_ref and atr_pct:
        floor = entry_ref * (1.0 - config.stop_atr_mult * atr_pct)
        if stop > floor:
            adjustments.append(f"stop widened {stop:.2f} → {floor:.2f} "
                               f"({config.stop_atr_mult:.1f}x ATR {atr_pct:.2%})")
            stop = floor
            out["stoploss"] = round(floor, 2)

    # Gate 4 — cost-adjusted reward/risk on the (possibly widened) geometry.
    tgt = _num(suggestion.get("sell_price"))
    if tgt is not None and stop is not None and entry_ref and entry_ref > 0:
        c_ = config.roundtrip_cost_pct
        reward = tgt / entry_ref - 1.0 - c_
        risk = 1.0 - stop / entry_ref + c_
        if risk <= 0 or reward / risk < config.min_eff_rr:
            eff = (reward / risk) if risk > 0 else float("-inf")
            reasons.append(f"cost-adj RR {eff:.2f} < {config.min_eff_rr:.1f}")

    if reasons:
        out["action"] = "HOLD"
        out["gate_reasons"] = reasons
        out["rationale"] = (str(suggestion.get("rationale") or "") +
                            " [entry gates: " + "; ".join(reasons) + "]").strip()
    if adjustments:
        out["gate_adjustments"] = adjustments
    return out


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


# ---------------------------------------------------------------------------
# Per-cycle entry cap (time-diversification)
# ---------------------------------------------------------------------------

def effective_rr(entry: Any, target: Any, stop: Any, cost_pct: float = 0.009) -> Optional[float]:
    """Cost-adjusted reward/risk — the same arithmetic as gate 4.

    ``(target% − cost) / (stop% + cost)`` on the ``entry`` reference price;
    ``None`` when any leg is missing/invalid or the risk term is non-positive.
    """
    e, t, s = _num(entry), _num(target), _num(stop)
    if e is None or t is None or s is None:
        return None
    reward = t / e - 1.0 - cost_pct
    risk = 1.0 - s / e + cost_pct
    if risk <= 0:
        return None
    return reward / risk


def apply_entry_cap(
    suggestions: Dict[str, Dict[str, Any]],
    held: Any,
    max_new: int,
    *,
    cost_pct: float = 0.009,
) -> List[str]:
    """Cap how many NEW positions may be opened this cycle (in place).

    The March-2022 loss cluster came from 5 correlated entries inside one 5-day
    window — diversification across names is no protection when every entry
    shares a timestamp. This caps fresh BUYs (names NOT already held) at
    ``max_new`` per cycle, keeping the highest-conviction ones (cost-adjusted
    RR, ties broken by LLM confidence) and demoting the overflow to HOLD with
    the deferral recorded in ``gate_reasons``. A deferred name is not lost: it
    re-qualifies at the next cycle against fresh forecasts. ``max_new <= 0``
    disables the cap. Returns the deferred tickers.
    """
    if not max_new or max_new <= 0:
        return []
    held_set = {str(t).upper() for t in (held or [])}
    fresh: List[Tuple[str, float, float]] = []
    for t, s in suggestions.items():
        if not isinstance(s, dict) or s.get("error"):
            continue
        if str(s.get("action", "")).upper() != "BUY" or str(t).upper() in held_set:
            continue
        entry_ref = s.get("buy_price") or s.get("last_close") or s.get("last_close_inr")
        rr = effective_rr(entry_ref, s.get("sell_price"), s.get("stoploss"), cost_pct)
        try:
            conf = float(s.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        fresh.append((t, rr if rr is not None else float("-inf"), conf))
    if len(fresh) <= max_new:
        return []
    fresh.sort(key=lambda x: (x[1], x[2]), reverse=True)
    deferred = [t for t, _, _ in fresh[max_new:]]
    for rank, t in enumerate(deferred, start=max_new + 1):
        s = suggestions[t]
        note = (f"entry cap: ranked {rank}/{len(fresh)} on cost-adj RR this cycle "
                f"(max {max_new} new positions) — deferred, re-qualifies next cycle")
        s["action"] = "HOLD"
        reasons = s.get("gate_reasons") or []
        s["gate_reasons"] = list(reasons) + [note]
        s["rationale"] = (str(s.get("rationale") or "") + f" [{note}]").strip()
    return deferred
