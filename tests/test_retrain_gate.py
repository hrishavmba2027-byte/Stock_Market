"""Unit tests for the incremental-retrain gate in monthly_finetune.py.

The gate decides when the automatic monthly fine-tune fires:
* calendar trigger — a new calendar month started since the last successful
  fine-tune and there is any new (untrained) data;
* data trigger — the median active symbol accumulated --min-new-rows rows
  the models were never trained on (early, mid-month trigger).
"""
from __future__ import annotations

from datetime import datetime, timezone

import monthly_finetune as mf

NOW = datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc)


def summary(symbol: str, rows: int, new_rows: int) -> mf.SymbolDatasetSummary:
    return mf.SymbolDatasetSummary(
        symbol=symbol,
        rows=rows,
        new_rows=new_rows,
        train_samples=0,
        validation_samples=0,
        replay_samples=0,
    )


def state_with_last_run(value: str) -> dict:
    return {"fine_tune": {"last_successful_run_utc": value}}


def test_never_ran_with_new_data_is_due_via_calendar():
    gate = mf.evaluate_retrain_gate({}, [summary("A", 100, 5), summary("B", 100, 7)], 30, NOW)
    assert gate["due"] is True
    assert gate["calendar_due"] is True
    assert gate["data_due"] is False
    assert gate["new_calendar_month_since_last_success"] is True


def test_same_month_below_threshold_is_not_due():
    state = state_with_last_run("2026-07-01T12:30:00Z")
    gate = mf.evaluate_retrain_gate(state, [summary("A", 100, 10), summary("B", 100, 20)], 30, NOW)
    assert gate["due"] is False
    assert gate["median_new_rows_per_active_symbol"] == 15.0


def test_new_month_with_any_new_data_is_due():
    state = state_with_last_run("2026-06-28T12:30:00Z")
    gate = mf.evaluate_retrain_gate(state, [summary("A", 100, 3), summary("B", 100, 3)], 30, NOW)
    assert gate["due"] is True
    assert gate["calendar_due"] is True


def test_same_month_median_reaching_threshold_fires_early():
    state = state_with_last_run("2026-07-01T12:30:00Z")
    summaries = [summary("A", 100, 31), summary("B", 100, 45), summary("C", 100, 29)]
    gate = mf.evaluate_retrain_gate(state, summaries, 30, NOW)
    assert gate["due"] is True
    assert gate["data_due"] is True
    assert gate["calendar_due"] is False


def test_new_month_but_zero_new_rows_is_not_due():
    state = state_with_last_run("2026-06-01T12:30:00Z")
    gate = mf.evaluate_retrain_gate(state, [summary("A", 100, 0), summary("B", 100, 0)], 30, NOW)
    assert gate["due"] is False


def test_stale_symbol_excluded_from_median():
    gate = mf.evaluate_retrain_gate({}, [summary("A", 0, 0), summary("B", 100, 40)], 30, NOW)
    assert gate["due"] is True
    assert gate["active_symbols"] == 1
    assert gate["median_new_rows_per_active_symbol"] == 40.0


def test_malformed_last_run_treated_as_never_ran():
    state = state_with_last_run("not-a-timestamp")
    gate = mf.evaluate_retrain_gate(state, [summary("A", 100, 2)], 30, NOW)
    assert gate["due"] is True
    assert gate["calendar_due"] is True


def test_no_symbols_is_not_due():
    gate = mf.evaluate_retrain_gate({}, [], 30, NOW)
    assert gate["due"] is False
    assert gate["active_symbols"] == 0


def test_parse_args_defaults_include_gate_flags(monkeypatch):
    monkeypatch.setattr("sys.argv", ["monthly_finetune.py"])
    monkeypatch.delenv("MIN_NEW_ROWS_FOR_FINETUNE", raising=False)
    args = mf.parse_args()
    assert args.min_new_rows == mf.DEFAULT_MIN_NEW_ROWS == 30
    assert args.if_due is False
    assert args.check_only is False


def test_min_new_rows_env_override(monkeypatch):
    monkeypatch.setattr("sys.argv", ["monthly_finetune.py"])
    monkeypatch.setenv("MIN_NEW_ROWS_FOR_FINETUNE", "45")
    args = mf.parse_args()
    assert args.min_new_rows == 45
