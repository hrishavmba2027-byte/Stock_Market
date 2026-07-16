"""Tests for the separate BacktestSettings config."""
from __future__ import annotations

from app.config.backtest_settings import BacktestSettings


def _clear(monkeypatch, *names):
    for n in names:
        monkeypatch.delenv(n, raising=False)


def test_defaults_are_inert(monkeypatch):
    _clear(monkeypatch, "BACKTEST_ENABLED", "BACKTEST_START_DATE", "BACKTEST_TICKERS",
           "BACKTEST_NEWS_INTERVAL_DAYS", "BACKTEST_HONOR_STOPLOSS")
    s = BacktestSettings.from_env()
    assert s.enabled is False
    assert s.start_date == "2015-01-01"
    assert s.news_interval_days == 7
    assert s.tickers == []
    # Fully local: news is read from a local workbook, and stop-loss exits are armed.
    assert str(s.news_workbook_path).endswith("backtest_news.xlsx")
    assert s.honor_stoploss is True


def test_env_overrides_are_backtest_scoped(monkeypatch):
    monkeypatch.setenv("BACKTEST_ENABLED", "true")
    monkeypatch.setenv("BACKTEST_START_DATE", "2020-01-01")
    monkeypatch.setenv("BACKTEST_END_DATE", "2021-12-31")
    monkeypatch.setenv("BACKTEST_TICKERS", "tcs, reliance")
    monkeypatch.setenv("BACKTEST_NEWS_INTERVAL_DAYS", "5")
    monkeypatch.setenv("BACKTEST_HONOR_STOPLOSS", "false")
    s = BacktestSettings.from_env()
    assert s.enabled is True
    assert s.tickers == ["TCS", "RELIANCE"]
    assert s.news_interval_days == 5
    assert s.honor_stoploss is False


def test_news_years_span(monkeypatch):
    monkeypatch.setenv("BACKTEST_START_DATE", "2019-06-01")
    monkeypatch.setenv("BACKTEST_END_DATE", "2022-03-01")
    s = BacktestSettings.from_env()
    assert s.news_years() == [2019, 2020, 2021, 2022]


def test_resolved_end_date_defaults_to_today(monkeypatch):
    monkeypatch.setenv("BACKTEST_END_DATE", "")
    s = BacktestSettings.from_env()
    from datetime import date

    assert s.resolved_end_date() == date.today().isoformat()


def test_backtest_model_dir_separate_from_production():
    """The backtest model must never share the production model dir."""
    from app.config.settings import get_settings

    bt = BacktestSettings.from_env()
    assert bt.model_dir.resolve() != get_settings().model_dir.resolve()
