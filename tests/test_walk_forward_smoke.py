"""End-to-end smoke test of the walk-forward loop with heavy pieces faked.

Fakes torch training/forecasting, the GLM call, and the workbook loader so the
whole event loop (PIT marks → next-bar fills → sizing → cash-guarded orders →
bookkeeping/checkpoint) runs offline in well under a second. Asserts the
invariants: cash never negative, equity finite, and the workbook is written with
every sheet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.backtest_settings import BacktestSettings
from backtesting import model_cycle, signals as bt_signals, walk_forward
import scripts.train_from_workbook as tfw


def _synthetic_frames(tickers, n=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    frames = {}
    for i, t in enumerate(tickers):
        steps = rng.normal(0.001, 0.02, n).cumsum()
        close = 100 * (1 + i * 0.1) * np.exp(steps)
        frames[t] = pd.DataFrame({
            "Date": dates,
            "Date_str": [d.strftime("%Y-%m-%d") for d in dates],
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1000, 5000, n),
        })
    return frames


@pytest.fixture()
def patched(monkeypatch, tmp_path):
    tickers = ["AAA", "BBB", "CCC"]
    frames = _synthetic_frames(tickers)

    monkeypatch.setattr(tfw, "build_frames", lambda workbook, tks: frames)
    monkeypatch.setattr(model_cycle, "seed_train_at", lambda *a, **k: {"mode": "seed"})
    monkeypatch.setattr(model_cycle, "retrain_at", lambda *a, **k: {"mode": "retrain"})

    def fake_forecast(fr, C, settings, **k):
        # Bullish +6% 15-day path off each name's last close ≤ C.
        out = {}
        for sym, f in fr.items():
            sub = f.loc[pd.to_datetime(f["Date"]) <= pd.Timestamp(C)]
            if len(sub) < 21:
                continue
            c = float(sub["Close"].iloc[-1])
            out[sym.upper()] = {f"T+{h}": c * (1 + 0.004 * h) for h in range(1, 16)}
        return out

    monkeypatch.setattr(model_cycle, "forecast_at", fake_forecast)

    def fake_run_signals(tks, forecasts, frames_asof, sent, funds, C, **k):
        res = {}
        for t in tks:
            fc = forecasts.get(t.upper())
            frame = frames_asof.get(t.upper())
            if not fc or frame is None or frame.empty:
                continue
            close = float(pd.to_numeric(frame["Close"], errors="coerce").dropna().iloc[-1])
            res[t.upper()] = {
                "as_of_date": pd.Timestamp(C).strftime("%Y-%m-%d"), "last_close": close,
                "action": "BUY", "buy_price": close, "sell_day": "T+10",
                "sell_price": close * 1.1, "stoploss": close * 0.97, "confidence": 0.7,
                "rationale": "synthetic", "sentiment_used": False, "fundamentals_used": False,
                "fingerprint": f"{t}-{pd.Timestamp(C).date()}",
            }
        return res

    monkeypatch.setattr(bt_signals, "run_signals", fake_run_signals)

    settings = BacktestSettings(
        enabled=True, tickers=tickers, start_date="2020-01-01", end_date="2020-03-01",
        initial_capital=1_000_000.0, retrain_interval_days=15, sentiment_interval_days=7,
        workbook_path=tmp_path / "wb.xlsx", state_path=tmp_path / "state.json",
        bookkeeping_path=tmp_path / "bk.xlsx", report_dir=tmp_path / "report",
        index_cache_path=tmp_path / "missing_indices.parquet",
        sentiment_cache_path=tmp_path / "missing_sent.parquet",
    )
    return settings


def test_walk_forward_runs_and_respects_cash_guard(patched):
    settings = patched
    result = walk_forward.run_backtest(settings, resume=False, device="cpu")

    assert result["bars_processed"] > 5
    assert np.isfinite(result["ending_equity"])

    # Reload state and assert the invariants over the whole run.
    from backtesting.state import BacktestState
    st = BacktestState.load_or_init(settings.state_path, initial_capital=1_000_000.0, resume=True)
    assert st.cash >= -1e-6
    for row in st.equity_curve:
        assert row["cash"] >= -1e-6
        assert row["equity"] > 0
    assert st.trade_log, "expected at least one fill (BUY queued at bar0, filled bar1)"
    assert any(t["side"] == "BUY" for t in st.trade_log)

    # Bookkeeping workbook written with every sheet (incl. the daily target/stop
    # monitor and the booked exits).
    xl = pd.ExcelFile(settings.bookkeeping_path)
    for sheet in ("LLM_Suggestions", "Allocations", "Trades", "EquityCurve", "PnL_Statement",
                  "Price_Monitor", "Exits"):
        assert sheet in xl.sheet_names
    alloc = pd.read_excel(settings.bookkeeping_path, sheet_name="Allocations")
    assert len(alloc) > 0

    # Held names carry an LLM target, so the daily price monitor logged rows and any
    # booked exit is a well-formed SELL settled at its trigger price.
    assert st.price_monitor_log, "expected daily target/stop monitoring rows"
    for ex in st.exit_log:
        assert ex["reason"] in ("target", "stop")
        assert ex["sale_amount"] > 0
    for t in st.trade_log:
        if str(t.get("reason", "")).startswith("exit:"):
            assert t["side"] == "SELL"

    # Report metrics emitted.
    assert (settings.report_dir / "metrics.json").exists()


def test_walk_forward_resume_is_idempotent(patched):
    settings = patched
    walk_forward.run_backtest(settings, resume=False, device="cpu", limit_bars=5)
    from backtesting.state import BacktestState
    st1 = BacktestState.load_or_init(settings.state_path, initial_capital=1_000_000.0, resume=True)
    bar1 = st1.last_bar
    # Resume: should continue past the checkpoint without reprocessing bar1.
    walk_forward.run_backtest(settings, resume=True, device="cpu", limit_bars=5)
    st2 = BacktestState.load_or_init(settings.state_path, initial_capital=1_000_000.0, resume=True)
    assert pd.Timestamp(st2.last_bar) >= pd.Timestamp(bar1)
    assert st2.cash >= -1e-6
