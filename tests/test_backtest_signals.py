"""Tests for backtesting.signals (reuses the production LLM layer; fake GLM)."""
from __future__ import annotations

import pandas as pd

from backtesting import signals as bt_signals
from features.trade_suggestions import _has_sentiment


def _frame():
    return pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=30, freq="D"),
        "Close": list(range(100, 130)),
    })


def test_build_market_shape():
    m = bt_signals.build_market("A", _frame(), {"T+1": 131, "T+15": 145}, "2020-01-30")
    assert m["date"] == "2020-01-30"
    assert m["close"] == 129
    from features.trade_suggestions import RECENT_CLOSES
    assert len(m["recent_closes"]) == min(RECENT_CLOSES, 30)   # capped at frame rows
    assert m["forecast"]["T+15"] == 145


def test_sentiment_gap_reaches_llm_as_unavailable():
    seen = {}

    def fake_ask(ticker, market, sentiment, model, api_key, fundamentals):
        # This is what production ask_glm receives; with no sentiment the context
        # has no company/sector/general data, so _has_sentiment must be False.
        seen[ticker] = _has_sentiment(sentiment) if isinstance(sentiment, dict) else False
        return {"action": "HOLD", "confidence": 0.5, "stoploss": 90}

    out = bt_signals.run_signals(
        ["RELIANCE"], {"RELIANCE": {"T+15": 150}}, {"RELIANCE": _frame()},
        sent_map={}, funds={}, cutoff="2020-01-30",
        model="fake", api_key="x", workers=1, ask=fake_ask,
    )
    assert out["RELIANCE"]["action"] == "HOLD"
    assert seen["RELIANCE"] is False        # sentiment gap -> "unavailable" downstream
    assert out["RELIANCE"]["sentiment_used"] is False


def test_fingerprint_skip_unchanged():
    calls = {"n": 0}

    def fake_ask(*a, **k):
        calls["n"] += 1
        return {"action": "BUY", "confidence": 0.7, "stoploss": 90}

    args = (["RELIANCE"], {"RELIANCE": {"T+15": 150}}, {"RELIANCE": _frame()})
    first = bt_signals.run_signals(*args, sent_map={}, funds={}, cutoff="2020-01-30",
                                   workers=1, ask=fake_ask)
    fp = first["RELIANCE"]["fingerprint"]
    # Re-run with the same fingerprint recorded as previous -> skipped, no LLM call.
    second = bt_signals.run_signals(*args, sent_map={}, funds={}, cutoff="2020-01-30",
                                    prev_fingerprints={"RELIANCE": fp}, workers=1, ask=fake_ask)
    assert calls["n"] == 1
    assert second == {}
