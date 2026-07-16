"""Unit tests for the incremental-retrain gate in monthly_finetune.py.

The gate decides when the automatic incremental fine-tune fires:
* interval trigger — RETRAIN_INTERVAL_DAYS (default 15) have elapsed since the
  last successful fine-tune and there is any new (untrained) data;
* data trigger — the median active symbol accumulated --min-new-rows rows
  the models were never trained on (early trigger, within the interval).
"""
from __future__ import annotations

from datetime import datetime, timezone

import monthly_finetune as mf

NOW = datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc)
INTERVAL = 15


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


def test_never_ran_with_new_data_is_due_via_interval():
    gate = mf.evaluate_retrain_gate({}, [summary("A", 100, 5), summary("B", 100, 7)], 30, INTERVAL, NOW)
    assert gate["due"] is True
    assert gate["interval_due"] is True
    assert gate["data_due"] is False
    assert gate["interval_elapsed_since_last_success"] is True
    assert gate["days_since_last_success"] is None


def test_within_interval_below_threshold_is_not_due():
    state = state_with_last_run("2026-07-05T12:30:00Z")  # ~7.9 days before NOW
    gate = mf.evaluate_retrain_gate(state, [summary("A", 100, 10), summary("B", 100, 20)], 30, INTERVAL, NOW)
    assert gate["due"] is False
    assert gate["interval_due"] is False
    assert gate["median_new_rows_per_active_symbol"] == 15.0


def test_interval_elapsed_with_any_new_data_is_due():
    state = state_with_last_run("2026-06-25T12:30:00Z")  # ~17.9 days before NOW
    gate = mf.evaluate_retrain_gate(state, [summary("A", 100, 3), summary("B", 100, 3)], 30, INTERVAL, NOW)
    assert gate["due"] is True
    assert gate["interval_due"] is True


def test_interval_boundary_exactly_elapsed_is_due():
    # Exactly 15 days before NOW → interval_elapsed uses >= so it fires.
    state = state_with_last_run("2026-06-28T10:00:00Z")
    gate = mf.evaluate_retrain_gate(state, [summary("A", 100, 2)], 30, INTERVAL, NOW)
    assert gate["days_since_last_success"] == 15.0
    assert gate["interval_due"] is True
    assert gate["due"] is True


def test_within_interval_median_reaching_threshold_fires_early():
    state = state_with_last_run("2026-07-08T12:30:00Z")  # ~4.9 days before NOW
    summaries = [summary("A", 100, 31), summary("B", 100, 45), summary("C", 100, 29)]
    gate = mf.evaluate_retrain_gate(state, summaries, 30, INTERVAL, NOW)
    assert gate["due"] is True
    assert gate["data_due"] is True
    assert gate["interval_due"] is False


def test_interval_elapsed_but_zero_new_rows_is_not_due():
    state = state_with_last_run("2026-06-01T12:30:00Z")  # well past 15 days
    gate = mf.evaluate_retrain_gate(state, [summary("A", 100, 0), summary("B", 100, 0)], 30, INTERVAL, NOW)
    assert gate["due"] is False
    assert gate["interval_elapsed_since_last_success"] is True


def test_stale_symbol_excluded_from_median():
    gate = mf.evaluate_retrain_gate({}, [summary("A", 0, 0), summary("B", 100, 40)], 30, INTERVAL, NOW)
    assert gate["due"] is True
    assert gate["active_symbols"] == 1
    assert gate["median_new_rows_per_active_symbol"] == 40.0


def test_malformed_last_run_treated_as_never_ran():
    state = state_with_last_run("not-a-timestamp")
    gate = mf.evaluate_retrain_gate(state, [summary("A", 100, 2)], 30, INTERVAL, NOW)
    assert gate["due"] is True
    assert gate["interval_due"] is True
    assert gate["days_since_last_success"] is None


def test_no_symbols_is_not_due():
    gate = mf.evaluate_retrain_gate({}, [], 30, INTERVAL, NOW)
    assert gate["due"] is False
    assert gate["active_symbols"] == 0


def test_parse_args_defaults_include_gate_flags(monkeypatch):
    monkeypatch.setattr("sys.argv", ["monthly_finetune.py"])
    monkeypatch.delenv("MIN_NEW_ROWS_FOR_FINETUNE", raising=False)
    monkeypatch.delenv("RETRAIN_INTERVAL_DAYS", raising=False)
    args = mf.parse_args()
    assert args.min_new_rows == mf.DEFAULT_MIN_NEW_ROWS == 30
    assert args.retrain_interval_days == mf.DEFAULT_RETRAIN_INTERVAL_DAYS == 15
    assert args.if_due is False
    assert args.check_only is False


def test_min_new_rows_env_override(monkeypatch):
    monkeypatch.setattr("sys.argv", ["monthly_finetune.py"])
    monkeypatch.setenv("MIN_NEW_ROWS_FOR_FINETUNE", "45")
    args = mf.parse_args()
    assert args.min_new_rows == 45


def test_retrain_interval_env_override(monkeypatch):
    monkeypatch.setattr("sys.argv", ["monthly_finetune.py"])
    monkeypatch.setenv("RETRAIN_INTERVAL_DAYS", "30")
    args = mf.parse_args()
    assert args.retrain_interval_days == 30
