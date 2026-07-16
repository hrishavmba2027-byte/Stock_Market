"""Round-trip tests for the cadence/duration config knobs in Settings.

Guards the audit contract: every cadence/duration must be a Settings field
sourced from env, not a hardcoded module constant.
"""
from __future__ import annotations

import pytest

from app.config.settings import Settings


def test_cadence_defaults():
    monkey_clear = {
        "NEWS_LOOKBACK_DAYS",
        "FUNDAMENTALS_REFRESH_DAYS",
        "RETRAIN_INTERVAL_DAYS",
        "SENTIMENT_HALFLIFE_DAYS",
    }
    import os

    saved = {k: os.environ.pop(k, None) for k in monkey_clear}
    try:
        s = Settings.from_env()
        assert s.news_lookback_days == 7
        assert s.fundamentals_refresh_days == 15
        assert s.retrain_interval_days == 15
        assert s.sentiment_halflife_days == 3.0
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_cadence_env_overrides(monkeypatch):
    monkeypatch.setenv("NEWS_LOOKBACK_DAYS", "10")
    monkeypatch.setenv("FUNDAMENTALS_REFRESH_DAYS", "21")
    monkeypatch.setenv("RETRAIN_INTERVAL_DAYS", "12")
    monkeypatch.setenv("SENTIMENT_HALFLIFE_DAYS", "5")
    s = Settings.from_env()
    assert s.news_lookback_days == 10
    assert s.fundamentals_refresh_days == 21
    assert s.retrain_interval_days == 12
    assert s.sentiment_halflife_days == 5.0


def test_cadence_minimums_enforced(monkeypatch):
    monkeypatch.setenv("NEWS_LOOKBACK_DAYS", "0")
    with pytest.raises(ValueError):
        Settings.from_env()
