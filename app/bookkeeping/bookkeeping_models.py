"""Dataclasses describing the bookkeeping input/output contract.

These types are the *stable* interface between Claude and the engine. They are
plain dataclasses (no third-party validation dependency) so the module stays
lightweight and portable.

Flow
----
Claude -> :class:`TradeRequest`  (parsed from JSON)
engine -> :class:`TradeResponse` (serialised back to JSON for Claude)
storage -> :class:`LedgerRow`    (one append-only audit row per evaluated order)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Outcome strings used across the ledger and the response.
DECISION_APPROVED = "APPROVED"
DECISION_PARTIAL = "PARTIAL"
DECISION_REJECTED = "REJECTED"

VALID_SIDES = {"buy", "sell"}


class BookkeepingError(ValueError):
    """Raised for malformed trade requests (bad input, not a system fault)."""


# --------------------------------------------------------------------------
# Input contract
# --------------------------------------------------------------------------
@dataclass
class Order:
    """A single proposed order line.

    ``error`` is set only by :meth:`parse_lenient` -- when populated the order
    is structurally invalid and the engine rejects it without raising, so one
    bad line never sinks an otherwise-valid portfolio batch.
    """

    symbol: str
    quantity: float
    price: float
    side: str = "buy"
    error: Optional[str] = None

    @classmethod
    def parse_lenient(cls, raw: Dict[str, Any], index: int = 0) -> "Order":
        """Parse an order, capturing any validation failure in ``error``."""
        try:
            return cls.from_dict(raw, index)
        except BookkeepingError as exc:
            symbol = ""
            if isinstance(raw, dict):
                symbol = str(raw.get("symbol", "")).strip().upper()

            def _safe_float(value: Any) -> float:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0

            return cls(
                symbol=symbol or f"ORDER_{index}",
                quantity=_safe_float(raw.get("quantity") if isinstance(raw, dict) else 0),
                price=_safe_float(raw.get("price") if isinstance(raw, dict) else 0),
                side=str(raw.get("side", "buy")).strip().lower() if isinstance(raw, dict) else "buy",
                error=str(exc),
            )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], index: int = 0) -> "Order":
        if not isinstance(raw, dict):
            raise BookkeepingError(f"Order #{index} must be an object, got {type(raw).__name__}.")

        symbol = str(raw.get("symbol", "")).strip().upper()
        if not symbol:
            raise BookkeepingError(f"Order #{index} is missing a 'symbol'.")

        if raw.get("quantity") is None:
            raise BookkeepingError(f"Order #{index} ({symbol}) is missing 'quantity'.")
        if raw.get("price") is None:
            raise BookkeepingError(f"Order #{index} ({symbol}) is missing 'price'.")

        try:
            quantity = float(raw["quantity"])
        except (TypeError, ValueError):
            raise BookkeepingError(
                f"Order #{index} ({symbol}) has a non-numeric 'quantity': {raw['quantity']!r}."
            )
        try:
            price = float(raw["price"])
        except (TypeError, ValueError):
            raise BookkeepingError(
                f"Order #{index} ({symbol}) has a non-numeric 'price': {raw['price']!r}."
            )

        side = str(raw.get("side", "buy")).strip().lower()
        if side not in VALID_SIDES:
            raise BookkeepingError(
                f"Order #{index} ({symbol}) has invalid side {side!r}; expected one of {sorted(VALID_SIDES)}."
            )

        if not math.isfinite(quantity) or not math.isfinite(price):
            raise BookkeepingError(f"Order #{index} ({symbol}) has non-finite quantity/price.")

        return cls(symbol=symbol, quantity=quantity, price=price, side=side)

    @property
    def order_value(self) -> float:
        return round(self.quantity * self.price, 2)


@dataclass
class TradeRequest:
    """A parsed, validated trade request from Claude."""

    trade_mode: str
    orders: List[Order]
    request_id: str
    side: str = "buy"
    dry_run: bool = False
    notes: str = ""
    timestamp: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TradeRequest":
        if not isinstance(raw, dict):
            raise BookkeepingError("Trade request must be a JSON object.")
        if not raw:
            raise BookkeepingError("Trade request is empty.")

        trade_mode = str(raw.get("trade_mode", "single")).strip().lower()
        if trade_mode not in {"single", "portfolio"}:
            raise BookkeepingError(
                f"Invalid trade_mode {trade_mode!r}; expected 'single' or 'portfolio'."
            )

        orders: List[Order] = []
        if trade_mode == "single":
            # A single request may use flat fields or a one-element 'orders' list.
            if "orders" in raw and raw["orders"]:
                orders = [Order.from_dict(raw["orders"][0], 0)]
            else:
                orders = [Order.from_dict(raw, 0)]
        else:
            order_list = raw.get("orders")
            # Also accept parallel symbols/quantities/prices arrays.
            if not order_list and "symbols" in raw:
                symbols = raw.get("symbols") or []
                quantities = raw.get("quantities") or []
                prices = raw.get("prices") or []
                if not (len(symbols) == len(quantities) == len(prices)):
                    raise BookkeepingError(
                        "Portfolio request: 'symbols', 'quantities' and 'prices' "
                        "must be arrays of equal length."
                    )
                order_list = [
                    {"symbol": s, "quantity": q, "price": p, "side": raw.get("side", "buy")}
                    for s, q, p in zip(symbols, quantities, prices)
                ]
            if not order_list:
                raise BookkeepingError("Portfolio request must include a non-empty 'orders' list.")
            if not isinstance(order_list, list):
                raise BookkeepingError("'orders' must be a list.")
            # Lenient: one malformed line is rejected individually, not fatally.
            orders = [Order.parse_lenient(o, i) for i, o in enumerate(order_list)]

        request_id = str(raw.get("request_id", "")).strip()
        if not request_id:
            # Deterministic fallback so the row is still auditable.
            request_id = "auto-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")

        timestamp = str(raw.get("timestamp", "")).strip() or datetime.now(timezone.utc).isoformat()

        return cls(
            trade_mode=trade_mode,
            orders=orders,
            request_id=request_id,
            side=str(raw.get("side", "buy")).strip().lower(),
            dry_run=bool(raw.get("dry_run", False)),
            notes=str(raw.get("notes", "")),
            timestamp=timestamp,
        )


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------
@dataclass
class OrderDecision:
    """The engine's verdict for one order line."""

    symbol: str
    side: str
    requested_quantity: float
    approved_quantity: float
    price: float
    decision: str
    reason: str
    order_value: float = 0.0
    max_quantity_allowed: float = 0.0
    realized_pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "requested_quantity": self.requested_quantity,
            "approved_quantity": self.approved_quantity,
            "price": self.price,
            "decision": self.decision,
            "reason": self.reason,
            "order_value": self.order_value,
            "max_quantity_allowed": self.max_quantity_allowed,
            "realized_pnl": self.realized_pnl,
        }


@dataclass
class LedgerRow:
    """One append-only audit row. Column order matches the Google Sheet."""

    date: str
    symbol: str
    qty: float
    buy_price: float
    order_value: float
    available_capital: float
    decision: str
    reason: str
    # Extended (kept after the eight core fields).
    request_id: str = ""
    side: str = "buy"
    trade_mode: str = "single"
    realized_pnl: float = 0.0
    capital_before: float = 0.0
    capital_after: float = 0.0

    COLUMNS = [
        "Date", "Symbol", "Qty", "Buy_Price", "Order_Value", "Available_Capital",
        "Decision", "Reason", "Request_ID", "Side", "Trade_Mode", "Realized_PnL",
        "Capital_Before", "Capital_After",
    ]

    def to_row(self) -> List[Any]:
        return [
            self.date, self.symbol, self.qty, self.buy_price, self.order_value,
            self.available_capital, self.decision, self.reason, self.request_id,
            self.side, self.trade_mode, self.realized_pnl, self.capital_before,
            self.capital_after,
        ]

    def to_dict_record(self) -> Dict[str, Any]:
        """Return a dict keyed by :data:`COLUMNS` for storage adapters."""
        return dict(zip(self.COLUMNS, self.to_row()))


@dataclass
class TradeResponse:
    """Structured result returned to Claude."""

    can_trade: bool
    trade_mode: str
    request_id: str
    reason: str
    approved_orders: List[Dict[str, Any]] = field(default_factory=list)
    rejected_orders: List[Dict[str, Any]] = field(default_factory=list)
    remaining_capital: float = 0.0
    max_quantity_allowed: float = 0.0
    net_order_value: float = 0.0
    capital_before: float = 0.0
    capital_after: float = 0.0
    realized_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    ledger_update_status: str = "not_written"
    dry_run: bool = False
    duplicate: bool = False
    currency: str = "INR"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "can_trade": self.can_trade,
            "trade_mode": self.trade_mode,
            "request_id": self.request_id,
            "reason": self.reason,
            "approved_orders": self.approved_orders,
            "rejected_orders": self.rejected_orders,
            "remaining_capital": self.remaining_capital,
            "max_quantity_allowed": self.max_quantity_allowed,
            "net_order_value": self.net_order_value,
            "capital_before": self.capital_before,
            "capital_after": self.capital_after,
            "realized_pnl": self.realized_pnl,
            "cumulative_pnl": self.cumulative_pnl,
            "ledger_update_status": self.ledger_update_status,
            "dry_run": self.dry_run,
            "duplicate": self.duplicate,
            "currency": self.currency,
            "warnings": self.warnings,
        }


@dataclass
class BookkeepingState:
    """The mutable, persisted state of the ledger.

    ``open_lots`` maps a symbol to a FIFO queue of open buy lots, each lot being
    ``{"qty": float, "price": float, "date": str, "request_id": str}``. It is the
    basis for realized-P&L calculation on sells.
    """

    initial_capital: float = 0.0
    available_capital: float = 0.0
    realized_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    open_lots: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    processed_request_ids: List[str] = field(default_factory=list)
    last_updated: str = ""
    trade_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "available_capital": self.available_capital,
            "realized_pnl": self.realized_pnl,
            "cumulative_pnl": self.cumulative_pnl,
            "open_lots": self.open_lots,
            "processed_request_ids": self.processed_request_ids,
            "last_updated": self.last_updated,
            "trade_count": self.trade_count,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BookkeepingState":
        return cls(
            initial_capital=float(raw.get("initial_capital", 0.0)),
            available_capital=float(raw.get("available_capital", 0.0)),
            realized_pnl=float(raw.get("realized_pnl", 0.0)),
            cumulative_pnl=float(raw.get("cumulative_pnl", 0.0)),
            open_lots=dict(raw.get("open_lots", {}) or {}),
            processed_request_ids=list(raw.get("processed_request_ids", []) or []),
            last_updated=str(raw.get("last_updated", "")),
            trade_count=int(raw.get("trade_count", 0)),
        )
