"""Walk-forward harness unit tests (pure modules; no network/LLM/torch)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from allocation.config import AllocationConfig
from backtesting import calendar as bt_calendar
from backtesting import costs as bt_costs
from backtesting import data_access as bt_data
from backtesting import portfolio as bt_portfolio
from backtesting.bookkeeping import BacktestBookkeeper
from backtesting.state import BacktestState


def cfg(**over) -> AllocationConfig:
    base = dict(initial_capital=1_000_000.0, cash_floor_frac=0.10, per_name_cap_frac=0.5,
                vol_target_annual=0.20, llm_confidence_tilt=0.25, min_ticket_inr=1_000.0,
                reallocation_edge=0.15)
    base.update(over)
    return AllocationConfig(**base)


def make_frame(dates, opens, closes):
    n = len(dates)
    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Open": opens,
        "High": [c * 1.01 for c in closes],
        "Low": [c * 0.99 for c in closes],
        "Close": closes,
        "Adj Close": closes,
        "Volume": [1000] * n,
    })


# --------------------------------------------------------------------------
# costs
# --------------------------------------------------------------------------
def test_roundtrip_cost_per_leg():
    assert bt_costs.roundtrip_cost_inr(100_000, bps=40) == pytest.approx(200.0)
    assert bt_costs.roundtrip_cost_inr(0) == 0.0


def test_atr_slippage():
    assert bt_costs.atr_slippage_inr(10, atr14=5.0, price=100, mult=0.10) == pytest.approx(5.0)
    # Bad ATR falls back to 1% of price (=1.0), *mult 0.1 *10sh = 1.0; never NaN/negative.
    val = bt_costs.atr_slippage_inr(10, atr14=float("nan"), price=100, mult=0.10)
    assert val == pytest.approx(1.0)


# --------------------------------------------------------------------------
# calendar
# --------------------------------------------------------------------------
def test_trading_days_union_and_bounds():
    frames = {
        "A": make_frame(["2020-01-01", "2020-01-02", "2020-01-06"], [1, 1, 1], [1, 1, 1]),
        "B": make_frame(["2020-01-02", "2020-01-03"], [1, 1], [1, 1]),
    }
    days = bt_calendar.trading_days(frames, "2020-01-02", "2020-01-06")
    assert days == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")]


def test_cadence_day0_both_and_retrain_supersedes():
    days = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i) for i in range(16)]
    sched = bt_calendar.cadence_schedule(days, retrain_every=15, sentiment_every=7)
    assert sched[0][1] == {"retrain": True}          # bar 0 = retrain (supersedes sentiment)
    assert sched[7][1] == {"sentiment": True}         # bar 7 = sentiment only
    assert sched[14][1] == {"sentiment": True}        # bar 14 = 7*2 sentiment
    assert sched[15][1] == {"retrain": True}          # bar 15 = retrain again
    assert sched[1][1] == {}                          # ordinary bar: no event


# --------------------------------------------------------------------------
# data_access — point-in-time
# --------------------------------------------------------------------------
def test_truncate_frames_max_date_is_cutoff():
    frames = {"A": make_frame(
        pd.date_range("2020-01-01", periods=10, freq="D"), [1] * 10, list(range(10, 20)))}
    out = bt_data.truncate_frames(frames, "2020-01-05")
    assert out["A"]["Date"].max() == pd.Timestamp("2020-01-05")
    assert (out["A"]["Date"] <= pd.Timestamp("2020-01-05")).all()


def test_returns_hist_asof_pit():
    frames = {"A": make_frame(
        pd.date_range("2020-01-01", periods=30, freq="D"), [1] * 30,
        list(np.linspace(100, 130, 30)))}
    hist = bt_data.returns_hist_asof(frames, "2020-01-10", lookback=252)
    # 10 closes -> 9 returns, all from <= cutoff
    assert len(hist["A"]) == 9


def test_open_on_returns_only_that_day():
    frames = {"A": make_frame(["2020-01-01", "2020-01-02"], [100, 200], [110, 210])}
    assert bt_data.open_on(frames, "2020-01-02") == {"A": 200.0}
    assert bt_data.open_on(frames, "2020-01-03") == {}


# --------------------------------------------------------------------------
# state — fills, no negative cash, realised P&L
# --------------------------------------------------------------------------
def test_state_buy_then_sell_realizes_pnl():
    st = BacktestState(cash=100_000, initial_capital=100_000)
    buy = bt_portfolio.Fill("A", "BUY", qty=100, price=100, gross=10_000, cost=20,
                            cash_delta=-10_020, reason="buy")
    st.apply_fills([buy], bar="2020-01-02")
    assert st.cash == pytest.approx(89_980)
    assert st.positions["A"]["qty"] == 100
    sell = bt_portfolio.Fill("A", "SELL", qty=100, price=120, gross=12_000, cost=24,
                             cash_delta=12_000 - 24, reason="sell")
    st.apply_fills([sell], bar="2020-01-10")
    assert "A" not in st.positions
    # realised = (120-100)*100 - 24 = 1976
    assert st.realized_pnl == pytest.approx(1976.0)
    assert st.cash >= 0


def test_state_never_goes_negative_on_overdraw():
    st = BacktestState(cash=5_000, initial_capital=5_000)
    bad = bt_portfolio.Fill("A", "BUY", qty=100, price=100, gross=10_000, cost=0,
                            cash_delta=-10_000, reason="overdraw")
    st.apply_fills([bad])
    assert st.cash >= 0
    assert "A" not in st.positions          # refused


def test_state_checkpoint_roundtrip(tmp_path: Path):
    st = BacktestState(cash=100_000, initial_capital=100_000, models_cutoff="2019-12-31")
    st.mark_to_market({"A": 100})
    st.record_equity("2020-01-01")
    p = tmp_path / "state.json"
    st.checkpoint(p)
    st2 = BacktestState.load_or_init(p, initial_capital=100_000, resume=True)
    assert st2.models_cutoff == "2019-12-31"
    assert st2.equity_curve[-1]["equity"] == pytest.approx(100_000)


# --------------------------------------------------------------------------
# portfolio — fill_at_open applies cost + slippage
# --------------------------------------------------------------------------
def test_fill_at_open_charges_costs():
    from allocation.engine import Order
    orders = [Order("A", "BUY", qty=100, price=0, notional=0, reason="x")]
    fills = bt_portfolio.fill_at_open(orders, {"A": 100.0}, atr_map={"A": 5.0},
                                      cost_bps=40, slippage_atr_mult=0.10)
    assert len(fills) == 1
    f = fills[0]
    # gross 10_000; txn 20; slippage 100*0.1*5=50 -> cost 70
    assert f.cost == pytest.approx(70.0)
    assert f.cash_delta == pytest.approx(-(10_000 + 70))


def test_fill_at_open_skips_missing_open():
    from allocation.engine import Order
    orders = [Order("A", "BUY", qty=100, price=0, notional=0, reason="x")]
    assert bt_portfolio.fill_at_open(orders, {}, atr_map={}) == []


# --------------------------------------------------------------------------
# bookkeeping — workbook has every sheet, P&L is coherent
# --------------------------------------------------------------------------
def test_bookkeeper_writes_all_sheets(tmp_path: Path):
    st = BacktestState(cash=100_000, initial_capital=100_000)
    st.apply_fills([bt_portfolio.Fill("A", "BUY", 100, 100, 10_000, 20, -10_020, "buy")], bar="2020-01-02")
    st.mark_to_market({"A": 120})
    st.record_equity("2020-01-03")
    bk = BacktestBookkeeper(tmp_path / "bk.xlsx")
    bk.suggestions.append({"bar": "2020-01-01", "ticker": "A", "action": "BUY"})
    out = bk.flush(st)
    assert out.exists()
    xl = pd.ExcelFile(out)
    for sheet in ("LLM_Suggestions", "Allocations", "Trades", "EquityCurve", "PnL_Statement", "Open_Positions"):
        assert sheet in xl.sheet_names
    pnl = pd.read_excel(out, sheet_name="PnL_Statement").set_index("metric")["value"]
    assert pnl["Initial capital"] == pytest.approx(100_000)
    # unrealised = (120-100)*100 = 2000
    assert pnl["Unrealized P&L"] == pytest.approx(2000.0)
