"""Pure post-hoc metrics for the walk-forward run.

Everything here is scoring-only: it may look at realized future bars (that is the
*point* of evaluation) but is never wired back into a decision. Inputs are the
plain equity curve / trade log / forecast log the state accumulates.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _equity_series(equity_curve: Sequence[Dict[str, Any]]) -> pd.Series:
    if not equity_curve:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r["bar"] for r in equity_curve])
    vals = [float(r["equity"]) for r in equity_curve]
    return pd.Series(vals, index=idx).sort_index()


def equity_metrics(equity_curve: Sequence[Dict[str, Any]], *, rf_annual: float = 0.0) -> Dict[str, float]:
    """CAGR, annualised Sharpe/Sortino, max drawdown and Calmar."""
    eq = _equity_series(equity_curve)
    if len(eq) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
                "calmar": 0.0, "total_return": 0.0, "volatility_annual": 0.0}
    rets = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)
    vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0
    rf_daily = rf_annual / TRADING_DAYS
    excess = rets - rf_daily
    sharpe = float(excess.mean() / rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if rets.std(ddof=1) > 0 else 0.0
    downside = rets[rets < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(excess.mean() / dstd * np.sqrt(TRADING_DAYS)) if dstd > 0 else 0.0
    max_dd = _max_drawdown(eq)
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    return {
        "cagr": cagr, "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_dd,
        "calmar": calmar, "total_return": total_return, "volatility_annual": vol,
    }


def _max_drawdown(eq: pd.Series) -> float:
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    return float(dd.min()) if len(dd) else 0.0


def benchmark_compare(
    equity_curve: Sequence[Dict[str, Any]],
    benchmark: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Alpha/beta and up/down capture of the strategy vs a benchmark series.

    ``benchmark`` is a price series indexed by date; if omitted, the per-bar
    ``benchmark`` field on the equity curve is used. Returns zeros when the two
    series don't overlap.
    """
    eq = _equity_series(equity_curve)
    if benchmark is None:
        bser = pd.Series(
            {pd.Timestamp(r["bar"]): r.get("benchmark") for r in equity_curve if r.get("benchmark") is not None}
        ).sort_index()
    else:
        bser = benchmark.sort_index()
    joined = pd.concat([eq.rename("strat"), bser.rename("bench")], axis=1).dropna()
    if len(joined) < 3:
        return {"alpha_annual": 0.0, "beta": 0.0, "up_capture": 0.0, "down_capture": 0.0}
    sr = joined["strat"].pct_change().dropna()
    br = joined["bench"].pct_change().dropna()
    common = sr.index.intersection(br.index)
    sr, br = sr.loc[common], br.loc[common]
    if len(sr) < 2 or br.var() == 0:
        return {"alpha_annual": 0.0, "beta": 0.0, "up_capture": 0.0, "down_capture": 0.0}
    beta = float(np.cov(sr, br)[0, 1] / np.var(br, ddof=1))
    alpha_daily = float(sr.mean() - beta * br.mean())
    up = br > 0
    down = br < 0
    up_cap = float(sr[up].mean() / br[up].mean()) if up.any() and br[up].mean() != 0 else 0.0
    down_cap = float(sr[down].mean() / br[down].mean()) if down.any() and br[down].mean() != 0 else 0.0
    return {
        "alpha_annual": alpha_daily * TRADING_DAYS,
        "beta": beta,
        "up_capture": up_cap,
        "down_capture": down_cap,
    }


def turnover_and_cost_drag(
    trade_log: Sequence[Dict[str, Any]],
    equity_curve: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """Total traded notional, average turnover, and cost drag vs ending equity."""
    traded = sum(abs(float(t.get("gross", 0.0))) for t in trade_log)
    costs = sum(float(t.get("cost", 0.0)) for t in trade_log)
    eq = _equity_series(equity_curve)
    avg_equity = float(eq.mean()) if len(eq) else 0.0
    ending = float(eq.iloc[-1]) if len(eq) else 0.0
    return {
        "total_traded_notional": traded,
        "total_costs": costs,
        "turnover_x": float(traded / avg_equity) if avg_equity else 0.0,
        "cost_drag_frac": float(costs / ending) if ending else 0.0,
        "n_trades": len(trade_log),
    }


def forecast_error(forecast_log: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Per-horizon MAE/RMSE of forecast vs realized close.

    Each forecast-log row is ``{ticker, bar, horizon, forecast, realized}`` (the
    realized close is joined in post-hoc by the harness). Rows without a realized
    value are ignored.
    """
    by_h: Dict[str, List[Any]] = {}
    for r in forecast_log:
        realized = r.get("realized")
        forecast = r.get("forecast")
        if realized is None or forecast is None:
            continue
        h = str(r.get("horizon"))
        by_h.setdefault(h, []).append((float(forecast), float(realized)))
    out: Dict[str, Dict[str, float]] = {}
    for h, pairs in by_h.items():
        arr = np.array(pairs, dtype=float)
        err = arr[:, 0] - arr[:, 1]
        out[h] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "n": int(len(err)),
        }
    return out


def action_conditioned_returns(trade_log: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Realized-P&L stats grouped by the reason/side of closed (SELL) trades."""
    sells = [t for t in trade_log if t.get("side") == "SELL"]
    if not sells:
        return {}
    pnls = np.array([float(t.get("realized_pnl", 0.0)) for t in sells])
    wins = pnls > 0
    return {
        "all_sells": {
            "n": int(len(pnls)),
            "mean_pnl": float(pnls.mean()),
            "total_pnl": float(pnls.sum()),
            "win_rate": float(wins.mean()),
        }
    }


def confidence_calibration(
    signal_log: Sequence[Dict[str, Any]],
    bins: int = 5,
) -> List[Dict[str, float]]:
    """Bucket BUY confidence vs realized outcome (needs joined ``realized_return``).

    Rows are ``{confidence, realized_return}``; returns per-bin mean confidence,
    mean realized return, and count — the calibration curve.
    """
    rows = [
        (float(r["confidence"]), float(r["realized_return"]))
        for r in signal_log
        if r.get("confidence") is not None and r.get("realized_return") is not None
    ]
    if not rows:
        return []
    arr = np.array(rows)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: List[Dict[str, float]] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (arr[:, 0] >= lo) & (arr[:, 0] < hi if i < bins - 1 else arr[:, 0] <= hi)
        if not mask.any():
            continue
        out.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "mean_confidence": float(arr[mask, 0].mean()),
            "mean_realized_return": float(arr[mask, 1].mean()),
            "n": int(mask.sum()),
        })
    return out
