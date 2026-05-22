"""Tests for the trade-capital bookkeeping engine.

Run with::

    pytest tests/test_bookkeeping.py -v

All tests use the ``local`` JSON backend against a temporary directory, so they
never touch Google Sheets or any network. The Google Sheets adapter is covered
at the *interface* level (it must satisfy the StorageAdapter contract).
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

from app.bookkeeping.bookkeeping import BookkeepingEngine
from app.bookkeeping.bookkeeping_config import BookkeepingConfig, load_config
from app.bookkeeping.bookkeeping_models import TradeRequest, BookkeepingError
from app.bookkeeping.bookkeeping_storage import (
    LocalJSONStorage,
    StorageAdapter,
    get_storage,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def make_config(tmp_path: Path, **overrides) -> BookkeepingConfig:
    cfg = BookkeepingConfig(
        backend="local",
        initial_capital=100_000.0,
        currency="INR",
        state_file=tmp_path / "state" / "bk_state.json",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture()
def engine(tmp_path: Path) -> BookkeepingEngine:
    cfg = make_config(tmp_path)
    eng = BookkeepingEngine(config=cfg, storage=get_storage(cfg))
    eng.init_backend()
    return eng


# --------------------------------------------------------------------------
# Config / environment loading
# --------------------------------------------------------------------------
def test_env_loading_defaults(monkeypatch):
    for var in [
        "BOOKKEEPING_BACKEND", "BOOKKEEPING_INITIAL_CAPITAL",
        "BOOKKEEPING_ALLOW_PARTIAL_FILLS", "BOOKKEEPING_DECISION_THRESHOLD",
    ]:
        monkeypatch.delenv(var, raising=False)
    cfg = load_config()
    assert cfg.backend == "google_sheets"
    assert cfg.initial_capital == 100_000.0
    assert cfg.allow_partial_fills is False
    assert cfg.decision_threshold == 1.0


def test_env_loading_overrides_and_validation(monkeypatch):
    monkeypatch.setenv("BOOKKEEPING_BACKEND", "local")
    monkeypatch.setenv("BOOKKEEPING_INITIAL_CAPITAL", "250000")
    monkeypatch.setenv("BOOKKEEPING_ALLOW_PARTIAL_FILLS", "true")
    monkeypatch.setenv("BOOKKEEPING_DECISION_THRESHOLD", "5")  # invalid -> clamps
    monkeypatch.setenv("BOOKKEEPING_MAX_SYMBOLS_PER_BATCH", "garbage")  # -> default
    cfg = load_config()
    assert cfg.backend == "local"
    assert cfg.initial_capital == 250_000.0
    assert cfg.allow_partial_fills is True
    assert cfg.decision_threshold == 1.0  # clamped
    assert cfg.max_symbols_per_batch == 20  # default after bad input
    assert any("DECISION_THRESHOLD" in w for w in cfg.warnings)


# --------------------------------------------------------------------------
# Capital validation & max-quantity
# --------------------------------------------------------------------------
def test_approve_when_capital_sufficient(engine):
    resp = engine.process({
        "trade_mode": "single", "symbol": "RELIANCE",
        "quantity": 10, "price": 2450.5, "request_id": "ok-1",
    })
    assert resp["can_trade"] is True
    assert resp["capital_before"] == 100_000.0
    assert resp["capital_after"] == round(100_000.0 - 10 * 2450.5, 2)
    assert resp["approved_orders"][0]["decision"] == "APPROVED"


def test_reject_when_capital_insufficient(engine):
    resp = engine.process({
        "trade_mode": "single", "symbol": "TCS",
        "quantity": 100, "price": 3799.0, "request_id": "rej-1",
    })
    assert resp["can_trade"] is False
    assert resp["rejected_orders"][0]["decision"] == "REJECTED"
    assert "insufficient capital" in resp["rejected_orders"][0]["reason"].lower()
    # Rejected trade must not move capital.
    assert resp["capital_after"] == 100_000.0


def test_max_quantity_calculation(engine):
    resp = engine.process({
        "trade_mode": "single", "symbol": "INFY",
        "quantity": 10_000, "price": 1500.0, "request_id": "maxq-1",
    })
    # 100000 / 1500 = 66.66 -> 66 affordable shares.
    assert resp["max_quantity_allowed"] == 66
    assert resp["can_trade"] is False  # full order rejected, partials off


def test_partial_fill_when_enabled(tmp_path):
    cfg = make_config(tmp_path, allow_partial_fills=True)
    eng = BookkeepingEngine(config=cfg, storage=get_storage(cfg))
    eng.init_backend()
    resp = eng.process({
        "trade_mode": "single", "symbol": "INFY",
        "quantity": 1000, "price": 1500.0, "request_id": "partial-1",
    })
    assert resp["can_trade"] is True
    approved = resp["approved_orders"][0]
    assert approved["decision"] == "PARTIAL"
    assert approved["approved_quantity"] == 66
    assert resp["capital_after"] == round(100_000.0 - 66 * 1500.0, 2)


# --------------------------------------------------------------------------
# Portfolio mode
# --------------------------------------------------------------------------
def test_portfolio_sequential_capital_depletion(engine):
    resp = engine.process({
        "trade_mode": "portfolio",
        "request_id": "batch-1",
        "orders": [
            {"symbol": "RELIANCE", "quantity": 20, "price": 2450.5},  # 49010
            {"symbol": "TCS", "quantity": 10, "price": 3799.0},       # 37990 -> ok (86k)
            {"symbol": "HDFC", "quantity": 50, "price": 1600.0},      # 80000 -> rejected
        ],
    })
    assert resp["trade_mode"] == "portfolio"
    assert len(resp["approved_orders"]) == 2
    assert len(resp["rejected_orders"]) == 1
    assert resp["rejected_orders"][0]["symbol"] == "HDFC"
    assert resp["capital_after"] == round(100_000.0 - 49010.0 - 37990.0, 2)


def test_portfolio_mixed_valid_and_invalid_orders(engine):
    resp = engine.process({
        "trade_mode": "portfolio",
        "request_id": "batch-mixed",
        "orders": [
            {"symbol": "RELIANCE", "quantity": 2, "price": 2450.5},
            {"symbol": "BADQTY", "price": 100.0},          # missing quantity
            {"symbol": "NEG", "quantity": -5, "price": 10.0},  # negative qty
        ],
    })
    assert len(resp["approved_orders"]) == 1
    assert len(resp["rejected_orders"]) == 2
    # The structurally-invalid order should still be auditable, not crash.
    symbols = {o["symbol"] for o in resp["rejected_orders"]}
    assert "BADQTY" in symbols and "NEG" in symbols


def test_portfolio_batch_size_limit(tmp_path):
    cfg = make_config(tmp_path, max_symbols_per_batch=2)
    eng = BookkeepingEngine(config=cfg, storage=get_storage(cfg))
    eng.init_backend()
    resp = eng.process({
        "trade_mode": "portfolio",
        "request_id": "too-big",
        "orders": [
            {"symbol": "A", "quantity": 1, "price": 1.0},
            {"symbol": "B", "quantity": 1, "price": 1.0},
            {"symbol": "C", "quantity": 1, "price": 1.0},
        ],
    })
    assert resp["can_trade"] is False
    assert "MAX_SYMBOLS_PER_BATCH" in resp["reason"]


# --------------------------------------------------------------------------
# Ledger writes & persistence
# --------------------------------------------------------------------------
def test_ledger_and_state_persistence(tmp_path):
    cfg = make_config(tmp_path)
    eng = BookkeepingEngine(config=cfg, storage=get_storage(cfg))
    eng.init_backend()
    eng.process({
        "trade_mode": "single", "symbol": "RELIANCE",
        "quantity": 10, "price": 2000.0, "request_id": "persist-1",
    })
    # A brand-new engine instance must see the persisted balance.
    eng2 = BookkeepingEngine(config=cfg, storage=get_storage(cfg))
    summary = eng2.get_state_summary()
    assert summary["available_capital"] == round(100_000.0 - 20_000.0, 2)
    assert summary["trade_count"] == 1
    assert summary["open_positions"]["RELIANCE"]["quantity"] == 10

    ledger = (tmp_path / "state" / "bookkeeping_ledger.jsonl").read_text().strip()
    assert "RELIANCE" in ledger
    executed = (tmp_path / "state" / "bookkeeping_executed.jsonl").read_text().strip()
    assert executed  # one executed row written


def test_dry_run_does_not_move_capital(engine):
    resp = engine.process({
        "trade_mode": "single", "symbol": "RELIANCE",
        "quantity": 5, "price": 1000.0, "request_id": "dry-1", "dry_run": True,
    })
    assert resp["dry_run"] is True
    assert resp["capital_after"] == resp["capital_before"] == 100_000.0
    assert resp["ledger_update_status"] == "suggestion_logged"
    assert engine.get_state_summary()["available_capital"] == 100_000.0


# --------------------------------------------------------------------------
# Idempotency / duplicate prevention
# --------------------------------------------------------------------------
def test_duplicate_request_id_is_not_reapplied(engine):
    first = engine.process({
        "trade_mode": "single", "symbol": "RELIANCE",
        "quantity": 10, "price": 2000.0, "request_id": "dup-1",
    })
    assert first["can_trade"] is True
    capital_after_first = first["capital_after"]

    second = engine.process({
        "trade_mode": "single", "symbol": "RELIANCE",
        "quantity": 10, "price": 2000.0, "request_id": "dup-1",
    })
    assert second["duplicate"] is True
    assert second["can_trade"] is False
    # Capital must be unchanged by the replay.
    assert engine.get_state_summary()["available_capital"] == capital_after_first


# --------------------------------------------------------------------------
# Realized P&L (FIFO, computed as trades execute)
# --------------------------------------------------------------------------
def test_realized_pnl_fifo_on_sell(engine):
    engine.process({
        "trade_mode": "single", "symbol": "TCS",
        "quantity": 10, "price": 1000.0, "request_id": "buy-lot-1",
    })
    engine.process({
        "trade_mode": "single", "symbol": "TCS",
        "quantity": 10, "price": 1200.0, "request_id": "buy-lot-2",
    })
    # Sell 15 @ 1500: 10 from lot1 (gain 500*10) + 5 from lot2 (gain 300*5).
    sell = engine.process({
        "trade_mode": "single", "symbol": "TCS", "side": "sell",
        "quantity": 15, "price": 1500.0, "request_id": "sell-1",
    })
    assert sell["can_trade"] is True
    assert sell["realized_pnl"] == 5000.0 + 1500.0  # 6500
    summary = engine.get_state_summary()
    assert summary["realized_pnl"] == 6500.0
    assert summary["cumulative_pnl"] == 6500.0
    assert summary["open_positions"]["TCS"]["quantity"] == 5


def test_sell_without_position_is_rejected(engine):
    resp = engine.process({
        "trade_mode": "single", "symbol": "WIPRO", "side": "sell",
        "quantity": 5, "price": 400.0, "request_id": "sell-bad",
    })
    assert resp["can_trade"] is False
    assert "no open position" in resp["rejected_orders"][0]["reason"].lower()


# --------------------------------------------------------------------------
# Edge cases & response formatting
# --------------------------------------------------------------------------
def test_empty_request_is_handled(engine):
    resp = engine.process({})
    assert resp["can_trade"] is False
    assert "invalid trade request" in resp["reason"].lower()


def test_missing_price_is_handled(engine):
    resp = engine.process({
        "trade_mode": "single", "symbol": "RELIANCE",
        "quantity": 10, "request_id": "no-price",
    })
    assert resp["can_trade"] is False


def test_invalid_trade_mode_is_handled(engine):
    resp = engine.process({
        "trade_mode": "telepathy", "symbol": "RELIANCE",
        "quantity": 1, "price": 1.0, "request_id": "bad-mode",
    })
    assert resp["can_trade"] is False


def test_zero_quantity_is_rejected(engine):
    resp = engine.process({
        "trade_mode": "single", "symbol": "RELIANCE",
        "quantity": 0, "price": 100.0, "request_id": "zero-qty",
    })
    assert resp["can_trade"] is False
    assert "positive" in resp["rejected_orders"][0]["reason"].lower()


def test_response_contains_full_contract(engine):
    resp = engine.process({
        "trade_mode": "single", "symbol": "RELIANCE",
        "quantity": 1, "price": 100.0, "request_id": "contract-1",
    })
    for key in [
        "can_trade", "approved_orders", "rejected_orders", "remaining_capital",
        "max_quantity_allowed", "reason", "net_order_value", "capital_before",
        "capital_after", "realized_pnl", "cumulative_pnl", "ledger_update_status",
        "trade_mode", "request_id",
    ]:
        assert key in resp, f"response missing contract key: {key}"
    # The whole response must be JSON-serialisable.
    json.dumps(resp)


# --------------------------------------------------------------------------
# Storage adapter interface
# --------------------------------------------------------------------------
def test_local_storage_satisfies_interface(tmp_path):
    cfg = make_config(tmp_path)
    storage = get_storage(cfg)
    assert isinstance(storage, StorageAdapter)
    assert isinstance(storage, LocalJSONStorage)
    ok, _ = storage.health_check()
    assert ok is True


def test_google_sheets_adapter_satisfies_interface():
    # Importable and structurally a StorageAdapter without needing credentials.
    from app.bookkeeping.bookkeeping_google_sheets import GoogleSheetsStorage

    assert issubclass(GoogleSheetsStorage, StorageAdapter)
    for method in ("init_schema", "load_state", "save_state",
                   "append_records", "health_check"):
        assert callable(getattr(GoogleSheetsStorage, method))


def test_portfolio_parallel_arrays_parsing():
    req = TradeRequest.from_dict({
        "trade_mode": "portfolio",
        "request_id": "arr-1",
        "symbols": ["RELIANCE", "TCS"],
        "quantities": [10, 4],
        "prices": [2450.5, 3799.0],
    })
    assert len(req.orders) == 2
    assert req.orders[0].symbol == "RELIANCE"
    assert req.orders[1].quantity == 4


def test_mismatched_parallel_arrays_raise():
    with pytest.raises(BookkeepingError):
        TradeRequest.from_dict({
            "trade_mode": "portfolio", "request_id": "arr-bad",
            "symbols": ["RELIANCE", "TCS"], "quantities": [10], "prices": [1.0],
        })
