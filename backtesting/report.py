"""Reporting: metrics JSON + charts for a completed walk-forward run.

Charts are best-effort (skipped cleanly if matplotlib is unavailable); the metrics
JSON is always written. Everything lands under ``settings.report_dir``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from backtesting import evaluate


def build_metrics(state: Any, *, benchmark: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Assemble the full metrics dict from the run's logs."""
    return {
        "equity": evaluate.equity_metrics(state.equity_curve),
        "benchmark": evaluate.benchmark_compare(state.equity_curve, benchmark),
        "turnover_cost": evaluate.turnover_and_cost_drag(state.trade_log, state.equity_curve),
        "forecast_error": evaluate.forecast_error(state.forecast_log),
        "action_conditioned": evaluate.action_conditioned_returns(state.trade_log),
        "confidence_calibration": evaluate.confidence_calibration(state.signal_log),
        "final": {
            "initial_capital": state.initial_capital,
            "ending_equity": round(state.equity(), 2),
            "realized_pnl": round(state.realized_pnl, 2),
            "total_costs": round(state.total_costs, 2),
            "n_trades": len(state.trade_log),
        },
    }


def write_report(state: Any, report_dir: Path, *, benchmark: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Write ``metrics.json`` and (if possible) the standard chart set."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(state, benchmark=benchmark)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    try:
        _write_charts(state, report_dir, benchmark, metrics)
    except Exception as exc:  # pragma: no cover - plotting is optional
        (report_dir / "charts_skipped.txt").write_text(f"charts skipped: {exc}\n")
    return metrics


def _write_charts(state: Any, report_dir: Path, benchmark: Optional[pd.Series], metrics: Dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not state.equity_curve:
        return
    eq = pd.DataFrame(state.equity_curve)
    eq["bar"] = pd.to_datetime(eq["bar"])
    eq = eq.set_index("bar").sort_index()

    # 1) Equity vs NIFTY (both rebased to the initial capital).
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq.index, eq["equity"], label="Strategy", color="#1f77b4")
    bench = benchmark if benchmark is not None else eq["benchmark"].dropna() if "benchmark" in eq else None
    if bench is not None and len(bench.dropna()) > 1:
        b = bench.dropna()
        rebased = b / b.iloc[0] * state.initial_capital
        ax.plot(rebased.index, rebased.values, label="NIFTY (rebased)", color="#ff7f0e", alpha=0.8)
    ax.set_title("Equity vs NIFTY (₹)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "equity_vs_nifty.png", dpi=120)
    plt.close(fig)

    # 2) Drawdown.
    running_max = eq["equity"].cummax()
    dd = eq["equity"] / running_max - 1.0
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    ax.set_title("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "drawdown.png", dpi=120)
    plt.close(fig)

    # 3) Forecast MAE by horizon.
    fe = metrics.get("forecast_error") or {}
    if fe:
        horizons = sorted(fe, key=lambda h: int(str(h).replace("T+", "")))
        mae = [fe[h]["mae"] for h in horizons]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(horizons)), mae, color="#2ca02c")
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels(horizons, rotation=45)
        ax.set_title("Forecast MAE by horizon (₹)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(report_dir / "forecast_mae_by_horizon.png", dpi=120)
        plt.close(fig)
