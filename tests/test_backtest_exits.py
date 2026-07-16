"""Target/stop intraday profit-booking — PIT-fair exit monitor + local news round-trip."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtesting import exits as bt_exits
from backtesting import fetch_news as fn
from backtesting import portfolio as bt_portfolio
from backtesting import sentiment_store


# --------------------------------------------------------------------------
# scan_exits — fill conventions never flatter the result
# --------------------------------------------------------------------------
def _pos(qty=100, avg=100.0):
    return {"qty": qty, "avg_price": avg}


def test_target_touched_books_at_target():
    positions = {"A": _pos()}
    sig = {"A": {"sell_price": 110.0, "stoploss": 90.0}}
    ohlc = {"A": {"open": 101, "high": 112, "low": 100, "close": 109}}   # high pierces target
    exits = bt_exits.scan_exits(positions, sig, ohlc)
    assert len(exits) == 1
    assert exits[0].reason == "target"
    assert exits[0].price == pytest.approx(110.0)          # limit fills at target


def test_target_gap_up_fills_at_open_not_above_range():
    positions = {"A": _pos()}
    sig = {"A": {"sell_price": 110.0, "stoploss": 90.0}}
    ohlc = {"A": {"open": 115, "high": 118, "low": 114, "close": 117}}   # gapped through target
    exits = bt_exits.scan_exits(positions, sig, ohlc)
    assert exits[0].reason == "target"
    assert exits[0].price == pytest.approx(115.0)          # better price at the open, not 110


def test_stop_touched_books_at_stop():
    positions = {"A": _pos()}
    sig = {"A": {"sell_price": 130.0, "stoploss": 95.0}}
    ohlc = {"A": {"open": 99, "high": 100, "low": 94, "close": 96}}      # low pierces stop
    exits = bt_exits.scan_exits(positions, sig, ohlc)
    assert exits[0].reason == "stop"
    assert exits[0].price == pytest.approx(95.0)


def test_stop_gap_down_fills_at_open_worse():
    positions = {"A": _pos()}
    sig = {"A": {"sell_price": 130.0, "stoploss": 95.0}}
    ohlc = {"A": {"open": 90, "high": 92, "low": 88, "close": 89}}       # gapped below stop
    exits = bt_exits.scan_exits(positions, sig, ohlc)
    assert exits[0].reason == "stop"
    assert exits[0].price == pytest.approx(90.0)            # worse fill at the gap open


def test_same_bar_ambiguity_resolves_to_stop():
    # One daily bar spans both stop and target; we cannot know order → assume stop.
    positions = {"A": _pos()}
    sig = {"A": {"sell_price": 110.0, "stoploss": 95.0}}
    ohlc = {"A": {"open": 100, "high": 112, "low": 94, "close": 105}}
    exits = bt_exits.scan_exits(positions, sig, ohlc)
    assert exits[0].reason == "stop"                        # pessimistic, no optimistic bias
    assert exits[0].price == pytest.approx(95.0)


def test_honor_stop_false_ignores_stop():
    positions = {"A": _pos()}
    sig = {"A": {"sell_price": 130.0, "stoploss": 95.0}}
    ohlc = {"A": {"open": 99, "high": 100, "low": 94, "close": 96}}      # only stop is touched
    assert bt_exits.scan_exits(positions, sig, ohlc, honor_stop=False) == []


def test_no_target_or_stop_no_exit():
    positions = {"A": _pos()}
    ohlc = {"A": {"open": 100, "high": 200, "low": 50, "close": 150}}
    assert bt_exits.scan_exits(positions, {"A": {}}, ohlc) == []


def test_untouched_range_no_exit():
    positions = {"A": _pos()}
    sig = {"A": {"sell_price": 130.0, "stoploss": 90.0}}
    ohlc = {"A": {"open": 100, "high": 105, "low": 98, "close": 102}}    # neither level reached
    assert bt_exits.scan_exits(positions, sig, ohlc) == []


def test_missing_bar_does_not_trigger():
    positions = {"A": _pos()}
    sig = {"A": {"sell_price": 110.0, "stoploss": 90.0}}
    assert bt_exits.scan_exits(positions, sig, ohlc={}) == []            # name didn't trade that day


# --------------------------------------------------------------------------
# fill_exits — full-position SELL with cost + slippage
# --------------------------------------------------------------------------
def test_fill_exits_liquidates_full_qty_with_costs():
    positions = {"A": _pos(qty=100)}
    exits = [bt_exits.ExitSignal("A", "target", price=110.0)]
    fills = bt_portfolio.fill_exits(exits, positions, atr_map={"A": 5.0},
                                    cost_bps=40, slippage_atr_mult=0.10)
    assert len(fills) == 1
    f = fills[0]
    assert f.side == "SELL" and f.qty == 100 and f.price == pytest.approx(110.0)
    # gross 11_000; txn 22; slippage 100*0.1*5=50 -> cost 72; cash_delta = gross - cost
    assert f.cost == pytest.approx(72.0)
    assert f.cash_delta == pytest.approx(11_000 - 72)
    assert f.reason == "exit:target"


def test_monitor_rows_flag_hits():
    positions = {"A": _pos(), "B": _pos()}
    sig = {"A": {"sell_price": 110.0, "stoploss": 90.0}, "B": {"sell_price": 130.0, "stoploss": 90.0}}
    ohlc = {"A": {"open": 101, "high": 112, "low": 100, "close": 109},
            "B": {"open": 100, "high": 105, "low": 98, "close": 102}}
    exits = bt_exits.scan_exits(positions, sig, ohlc)
    rows = bt_exits.monitor_rows("2020-02-03", positions, sig, ohlc, exits)
    by = {r["ticker"]: r for r in rows}
    assert by["A"]["hit"] == "target" and by["A"]["exit_price"] == pytest.approx(110.0)
    assert by["B"]["hit"] == "" and by["B"]["exit_price"] is None


# --------------------------------------------------------------------------
# Local news workbook → sentiment reader round-trip (no Firestore), strict PIT
# --------------------------------------------------------------------------
def test_workbook_sentiment_is_read_point_in_time(tmp_path: Path):
    wb = tmp_path / "news.xlsx"
    target = {"id": "TCS", "kind": "company", "display_name": "TCS"}
    # Two windows: one visible at the cutoff, one strictly after it.
    fn.store_to_workbook(wb, target, 2020,
                         [{"url": "https://a", "title": "t", "article_date": "2020-01-05"}],
                         {"2020-01-07": {"mean_sentiment": 0.3, "n": 4, "pos_share": 0.7, "neg_share": 0.1}})
    fn.store_to_workbook(wb, target, 2020,
                         [{"url": "https://b", "title": "t2", "article_date": "2020-01-20"}],
                         {"2020-01-21": {"mean_sentiment": -0.4, "n": 6, "pos_share": 0.1, "neg_share": 0.7}})

    # As of 2020-01-10 with a 7-day window, only the 2020-01-07 snapshot is visible.
    sent = sentiment_store.window_asof(["TCS"], "2020-01-10", window_days=7, workbook_path=wb)
    assert "TCS" in sent
    assert len(sent["TCS"]) == 1
    assert sent["TCS"][0]["as_of_date"] == "2020-01-07"
    assert sent["TCS"][0]["sent_mean_7d"] == pytest.approx(0.3)

    # The future window (2020-01-21) never leaks into an earlier decision.
    later = sentiment_store.window_asof(["TCS"], "2020-01-10", window_days=30, workbook_path=wb)
    dates = {s["as_of_date"] for s in later.get("TCS", [])}
    assert "2020-01-21" not in dates


def test_window_asof_absent_when_no_local_source():
    # No workbook, no cache → entity simply absent (reads as "unavailable" downstream).
    assert sentiment_store.window_asof(["TCS"], "2020-01-10", workbook_path=None, cache_path=None) == {}
