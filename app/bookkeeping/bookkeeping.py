"""The bookkeeping decision engine.

:class:`BookkeepingEngine` answers a single question for Claude:

    *Given the available cash balance, the proposed stock(s), order
    quantity(ies) and price per share -- can this trade be taken?*

It then (optionally) records the outcome in a complete, auditable ledger and
returns a structured result that Claude uses to decide what to recommend next.

Design notes
------------
* **Cash-only by default.** Capital logic lives in small, named helpers so a
  margin / blocked-funds / leverage mode can be layered on later without
  touching the decision flow.
* **Storage-agnostic.** The engine only ever talks to a
  :class:`~app.bookkeeping.bookkeeping_storage.StorageAdapter`.
* **Idempotent.** A non-dry-run ``request_id`` is recorded once; replaying it
  never double-spends capital.
* **Audit-first.** Every evaluated order produces a ``Ledger`` row whether it
  is approved or rejected.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.bookkeeping.bookkeeping_config import BookkeepingConfig, load_config
from app.bookkeeping.bookkeeping_models import (
    DECISION_APPROVED,
    DECISION_PARTIAL,
    DECISION_REJECTED,
    BookkeepingError,
    BookkeepingState,
    LedgerRow,
    Order,
    OrderDecision,
    TradeRequest,
    TradeResponse,
)
from app.bookkeeping.bookkeeping_storage import StorageAdapter, get_storage

logger = logging.getLogger("bookkeeping.engine")

EPSILON = 1e-9
MAX_TRACKED_REQUEST_IDS = 5000


class BookkeepingEngine:
    """Stateful trade-capital control engine."""

    def __init__(
        self,
        config: Optional[BookkeepingConfig] = None,
        storage: Optional[StorageAdapter] = None,
    ) -> None:
        self.config = config or load_config()
        self.storage = storage or get_storage(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def init_backend(self) -> Tuple[bool, str]:
        """Provision the backend schema. Safe to call repeatedly."""
        self.storage.init_schema()
        return self.storage.health_check()

    def health_check(self) -> Dict[str, Any]:
        ok, detail = self.storage.health_check()
        return {
            "backend": self.config.backend,
            "storage": self.storage.describe(),
            "healthy": ok,
            "detail": detail,
            "config": self.config.as_public_dict(),
            "config_warnings": self.config.warnings,
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a read-only snapshot of the current ledger state."""
        state = self.storage.load_state()
        positions = {
            symbol: {
                "quantity": round(sum(l["qty"] for l in lots), 4),
                "avg_price": round(
                    sum(l["qty"] * l["price"] for l in lots)
                    / max(sum(l["qty"] for l in lots), EPSILON),
                    4,
                ),
            }
            for symbol, lots in state.open_lots.items()
            if lots
        }
        return {
            "initial_capital": state.initial_capital,
            "available_capital": state.available_capital,
            "realized_pnl": state.realized_pnl,
            "cumulative_pnl": state.cumulative_pnl,
            "trade_count": state.trade_count,
            "open_positions": positions,
            "currency": self.config.currency,
            "last_updated": state.last_updated,
        }

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate (and, unless ``dry_run``, execute) a trade request.

        ``request`` is the raw dict from Claude. The return value is the
        structured response dict described in the I/O contract.
        """
        # 1. Parse / validate the request envelope.
        try:
            parsed = TradeRequest.from_dict(request)
        except BookkeepingError as exc:
            logger.warning("Rejected malformed trade request: %s", exc)
            return TradeResponse(
                can_trade=False,
                trade_mode=str(request.get("trade_mode", "single")) if isinstance(request, dict) else "single",
                request_id=str(request.get("request_id", "")) if isinstance(request, dict) else "",
                reason=f"Invalid trade request: {exc}",
                ledger_update_status="skipped",
                currency=self.config.currency,
            ).to_dict()

        # 2. Load current state.
        try:
            state = self.storage.load_state()
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not load bookkeeping state: %s", exc)
            return TradeResponse(
                can_trade=False,
                trade_mode=parsed.trade_mode,
                request_id=parsed.request_id,
                reason=f"Storage unavailable: {exc}",
                ledger_update_status="error",
                currency=self.config.currency,
            ).to_dict()

        # 3. Idempotency -- never re-apply an already-executed request.
        if not parsed.dry_run and parsed.request_id in set(state.processed_request_ids):
            logger.info("Duplicate request_id %s ignored.", parsed.request_id)
            return TradeResponse(
                can_trade=False,
                trade_mode=parsed.trade_mode,
                request_id=parsed.request_id,
                reason=(
                    f"Duplicate request_id '{parsed.request_id}' was already "
                    "executed; no capital was changed."
                ),
                remaining_capital=state.available_capital,
                capital_before=state.available_capital,
                capital_after=state.available_capital,
                realized_pnl=0.0,
                cumulative_pnl=state.cumulative_pnl,
                ledger_update_status="skipped_duplicate",
                duplicate=True,
                dry_run=parsed.dry_run,
                currency=self.config.currency,
            ).to_dict()

        # 4. Batch-size guard for portfolio requests.
        if (
            parsed.trade_mode == "portfolio"
            and len(parsed.orders) > self.config.max_symbols_per_batch
        ):
            return TradeResponse(
                can_trade=False,
                trade_mode=parsed.trade_mode,
                request_id=parsed.request_id,
                reason=(
                    f"Portfolio request has {len(parsed.orders)} orders, exceeding "
                    f"BOOKKEEPING_MAX_SYMBOLS_PER_BATCH={self.config.max_symbols_per_batch}."
                ),
                ledger_update_status="skipped",
                currency=self.config.currency,
            ).to_dict()

        # 5. Evaluate every order against capital (no mutation yet).
        capital_before = round(state.available_capital, 2)
        decisions = self._evaluate_orders(parsed, state)

        approved = [d for d in decisions if d.decision in (DECISION_APPROVED, DECISION_PARTIAL)]
        rejected = [d for d in decisions if d.decision == DECISION_REJECTED]

        # net_order_value: capital deployed by buys minus proceeds from sells.
        net_order_value = round(
            sum(
                d.approved_quantity * d.price * (1 if d.side == "buy" else -1)
                for d in approved
            ),
            2,
        )
        request_realized_pnl = round(sum(d.realized_pnl for d in approved), 2)
        can_trade = len(approved) > 0
        capital_after = round(capital_before - net_order_value, 2)

        # 6. Persist: dry-run logs suggestions; execution mutates state + ledger.
        ledger_status = "not_written"
        warnings: List[str] = list(self.config.warnings)

        if parsed.dry_run:
            ledger_status = self._record_suggestions(parsed, decisions, warnings)
            capital_after = capital_before  # nothing committed
            request_realized_pnl = 0.0
        elif can_trade or rejected:
            ledger_status = self._commit(
                parsed, decisions, state, capital_before, warnings
            )
            capital_after = round(state.available_capital, 2)
            # request_realized_pnl was computed pre-commit from the same
            # decisions that _commit applied, so it is already correct.

        # 7. Always log the request/response for traceability.
        self._safe_log_request(parsed, decisions, can_trade, warnings)

        # 8. Build the structured response for Claude.
        response = TradeResponse(
            can_trade=can_trade,
            trade_mode=parsed.trade_mode,
            request_id=parsed.request_id,
            reason=self._summarise_reason(parsed, approved, rejected),
            approved_orders=[d.to_dict() for d in approved],
            rejected_orders=[d.to_dict() for d in rejected],
            remaining_capital=capital_after,
            max_quantity_allowed=self._headline_max_qty(parsed, decisions),
            net_order_value=net_order_value,
            capital_before=capital_before,
            capital_after=capital_after,
            realized_pnl=request_realized_pnl,
            cumulative_pnl=round(state.cumulative_pnl, 2),
            ledger_update_status=ledger_status,
            dry_run=parsed.dry_run,
            currency=self.config.currency,
            warnings=warnings,
        )
        return response.to_dict()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate_orders(
        self, request: TradeRequest, state: BookkeepingState
    ) -> List[OrderDecision]:
        """Decide each order against a running capital pool.

        Portfolio orders are assessed sequentially: each approved buy reduces
        the pool available to subsequent orders, each approved sell adds its
        proceeds back. This is the cash-only capital model -- the single seam a
        future margin mode would extend.
        """
        # Spendable pool = available capital * decision threshold (cash reserve).
        running_capital = round(state.available_capital * self.config.decision_threshold, 6)
        # Lots are *copied* so evaluation never mutates persisted state.
        sim_lots: Dict[str, List[Dict[str, Any]]] = {
            sym: [dict(lot) for lot in lots] for sym, lots in state.open_lots.items()
        }
        decisions: List[OrderDecision] = []

        for order in request.orders:
            decision = self._evaluate_one(order, running_capital, sim_lots)
            decisions.append(decision)
            if decision.decision in (DECISION_APPROVED, DECISION_PARTIAL):
                if decision.side == "buy":
                    running_capital -= decision.approved_quantity * decision.price
                else:  # sell frees capital
                    running_capital += decision.approved_quantity * decision.price
        return decisions

    def _evaluate_one(
        self,
        order: Order,
        running_capital: float,
        sim_lots: Dict[str, List[Dict[str, Any]]],
    ) -> OrderDecision:
        base = OrderDecision(
            symbol=order.symbol,
            side=order.side,
            requested_quantity=order.quantity,
            approved_quantity=0.0,
            price=order.price,
            decision=DECISION_REJECTED,
            reason="",
            order_value=round(order.quantity * order.price, 2),
        )

        # Structural error captured during lenient parsing.
        if order.error:
            base.reason = order.error
            return base

        # Value validation.
        if order.quantity <= 0:
            base.reason = f"Quantity must be positive (got {order.quantity})."
            return base
        if order.price <= 0:
            base.reason = f"Price must be positive (got {order.price})."
            return base
        if not math.isfinite(order.quantity) or not math.isfinite(order.price):
            base.reason = "Quantity and price must be finite numbers."
            return base

        if order.side == "sell":
            return self._evaluate_sell(order, sim_lots, base)
        return self._evaluate_buy(order, running_capital, base)

    def _evaluate_buy(
        self, order: Order, running_capital: float, base: OrderDecision
    ) -> OrderDecision:
        order_value = round(order.quantity * order.price, 2)
        max_qty = math.floor(max(running_capital, 0.0) / order.price) if order.price > 0 else 0
        base.order_value = order_value
        base.max_quantity_allowed = float(max_qty)

        if order_value <= running_capital + EPSILON:
            base.decision = DECISION_APPROVED
            base.approved_quantity = order.quantity
            base.reason = (
                f"Approved: order value {order_value} <= available "
                f"{round(running_capital, 2)} {self.config.currency}."
            )
        elif self.config.allow_partial_fills and max_qty >= 1:
            base.decision = DECISION_PARTIAL
            base.approved_quantity = float(max_qty)
            base.order_value = round(max_qty * order.price, 2)
            base.reason = (
                f"Partial fill: capital allows {max_qty} of {order.quantity} "
                f"shares (need {order_value}, have {round(running_capital, 2)} "
                f"{self.config.currency})."
            )
        else:
            base.decision = DECISION_REJECTED
            base.approved_quantity = 0.0
            base.reason = (
                f"Rejected: insufficient capital -- need {order_value} "
                f"{self.config.currency} but only {round(running_capital, 2)} "
                f"available (max affordable quantity {max_qty}). Partial fills "
                f"are {'enabled' if self.config.allow_partial_fills else 'disabled'}."
            )
        return base

    def _evaluate_sell(
        self,
        order: Order,
        sim_lots: Dict[str, List[Dict[str, Any]]],
        base: OrderDecision,
    ) -> OrderDecision:
        lots = sim_lots.get(order.symbol, [])
        held = sum(lot["qty"] for lot in lots)
        base.max_quantity_allowed = float(held)

        if held <= EPSILON:
            base.reason = f"Rejected: no open position in {order.symbol} to sell."
            return base

        sell_qty = order.quantity
        partial = False
        if order.quantity > held + EPSILON:
            if self.config.allow_partial_fills:
                sell_qty = held
                partial = True
            else:
                base.reason = (
                    f"Rejected: sell quantity {order.quantity} exceeds held "
                    f"{round(held, 4)} in {order.symbol}; partial fills disabled."
                )
                return base

        realized = self._fifo_consume(lots, sell_qty, order.price)
        base.decision = DECISION_PARTIAL if partial else DECISION_APPROVED
        base.approved_quantity = sell_qty
        base.order_value = round(sell_qty * order.price, 2)
        base.realized_pnl = round(realized, 2)
        base.reason = (
            f"{'Partial ' if partial else ''}Sell approved: {sell_qty} share(s) "
            f"of {order.symbol} at {order.price}; realized P&L {base.realized_pnl} "
            f"{self.config.currency}."
        )
        # Reflect consumption in the simulation copy for subsequent orders.
        sim_lots[order.symbol] = [lot for lot in lots if lot["qty"] > EPSILON]
        return base

    @staticmethod
    def _fifo_consume(
        lots: List[Dict[str, Any]], sell_qty: float, sell_price: float
    ) -> float:
        """Consume ``sell_qty`` from ``lots`` FIFO, mutating lots in place.

        Returns realized P&L = sum((sell_price - lot_price) * matched_qty).
        """
        remaining = sell_qty
        realized = 0.0
        for lot in lots:
            if remaining <= EPSILON:
                break
            matched = min(lot["qty"], remaining)
            realized += (sell_price - lot["price"]) * matched
            lot["qty"] -= matched
            remaining -= matched
        return realized

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _commit(
        self,
        request: TradeRequest,
        decisions: List[OrderDecision],
        state: BookkeepingState,
        capital_before: float,
        warnings: List[str],
    ) -> str:
        """Apply approved orders to ``state`` and write the ledger."""
        running_capital = state.available_capital
        ledger_rows: List[LedgerRow] = []
        executed_rows: List[LedgerRow] = []
        rejected_rows: List[LedgerRow] = []
        now = self._now_str()

        for decision in decisions:
            cap_before_line = round(running_capital, 2)
            if decision.decision in (DECISION_APPROVED, DECISION_PARTIAL):
                if decision.side == "buy":
                    running_capital -= decision.approved_quantity * decision.price
                    state.open_lots.setdefault(decision.symbol, []).append({
                        "qty": decision.approved_quantity,
                        "price": decision.price,
                        "date": now,
                        "request_id": request.request_id,
                    })
                else:  # sell
                    running_capital += decision.approved_quantity * decision.price
                    self._fifo_consume(
                        state.open_lots.get(decision.symbol, []),
                        decision.approved_quantity,
                        decision.price,
                    )
                    state.open_lots[decision.symbol] = [
                        lot for lot in state.open_lots.get(decision.symbol, [])
                        if lot["qty"] > EPSILON
                    ]
                    state.realized_pnl = round(state.realized_pnl + decision.realized_pnl, 2)
                state.trade_count += 1

            row = LedgerRow(
                date=now,
                symbol=decision.symbol,
                qty=decision.approved_quantity,
                buy_price=decision.price,
                order_value=decision.order_value,
                available_capital=round(running_capital, 2),
                decision=decision.decision,
                reason=decision.reason,
                request_id=request.request_id,
                side=decision.side,
                trade_mode=request.trade_mode,
                realized_pnl=decision.realized_pnl,
                capital_before=cap_before_line,
                capital_after=round(running_capital, 2),
            )
            ledger_rows.append(row)
            if decision.decision in (DECISION_APPROVED, DECISION_PARTIAL):
                executed_rows.append(row)
            else:
                rejected_rows.append(row)

        # Update state aggregates.
        state.available_capital = round(running_capital, 2)
        state.cumulative_pnl = round(state.realized_pnl, 2)
        state.processed_request_ids.append(request.request_id)
        if len(state.processed_request_ids) > MAX_TRACKED_REQUEST_IDS:
            state.processed_request_ids = state.processed_request_ids[-MAX_TRACKED_REQUEST_IDS:]

        # Write ledger + sections, then state. Failures are reported, not fatal.
        status = "written"
        try:
            self.storage.append_records(
                "ledger", [r.to_dict_record() for r in ledger_rows]
            )
            if executed_rows:
                self.storage.append_records(
                    "executed", [r.to_dict_record() for r in executed_rows]
                )
            if rejected_rows:
                self.storage.append_records(
                    "rejected", [r.to_dict_record() for r in rejected_rows]
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Ledger write failed: %s", exc)
            warnings.append(f"Ledger write failed: {exc}")
            status = "ledger_write_failed"

        try:
            self.storage.save_state(state)
        except Exception as exc:  # noqa: BLE001
            logger.error("State save failed: %s", exc)
            warnings.append(f"State save failed: {exc}")
            status = "state_write_failed"

        return status

    def _record_suggestions(
        self,
        request: TradeRequest,
        decisions: List[OrderDecision],
        warnings: List[str],
    ) -> str:
        """Persist dry-run evaluations as open suggestions."""
        now = self._now_str()
        records = [
            {
                "Date": now,
                "Request_ID": request.request_id,
                "Symbol": d.symbol,
                "Side": d.side,
                "Qty": d.requested_quantity,
                "Price": d.price,
                "Order_Value": d.order_value,
                "Trade_Mode": request.trade_mode,
                "Note": f"{d.decision}: {d.reason}",
            }
            for d in decisions
        ]
        try:
            self.storage.append_records("suggestions", records)
            return "suggestion_logged"
        except Exception as exc:  # noqa: BLE001
            logger.error("Suggestion write failed: %s", exc)
            warnings.append(f"Suggestion write failed: {exc}")
            return "suggestion_write_failed"

    def _safe_log_request(
        self,
        request: TradeRequest,
        decisions: List[OrderDecision],
        can_trade: bool,
        warnings: List[str],
    ) -> None:
        import json

        record = {
            "Date": self._now_str(),
            "Request_ID": request.request_id,
            "Trade_Mode": request.trade_mode,
            "Dry_Run": request.dry_run,
            "Can_Trade": can_trade,
            "Request_JSON": json.dumps(
                {
                    "trade_mode": request.trade_mode,
                    "request_id": request.request_id,
                    "dry_run": request.dry_run,
                    "notes": request.notes,
                    "orders": [
                        {"symbol": o.symbol, "quantity": o.quantity,
                         "price": o.price, "side": o.side}
                        for o in request.orders
                    ],
                },
                default=str,
            ),
            "Response_JSON": json.dumps([d.to_dict() for d in decisions], default=str),
        }
        try:
            self.storage.append_records("request_log", [record])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Request-log write failed (non-fatal): %s", exc)
            warnings.append(f"Request-log write failed: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _headline_max_qty(
        self, request: TradeRequest, decisions: List[OrderDecision]
    ) -> float:
        """Top-level max_quantity_allowed.

        For single requests this is the one order's affordable quantity. For
        portfolio requests it is the total approved quantity (per-order limits
        are in each entry of approved_orders / rejected_orders).
        """
        if not decisions:
            return 0.0
        if request.trade_mode == "single":
            return decisions[0].max_quantity_allowed
        return round(
            sum(
                d.approved_quantity
                for d in decisions
                if d.decision in (DECISION_APPROVED, DECISION_PARTIAL)
            ),
            4,
        )

    @staticmethod
    def _summarise_reason(
        request: TradeRequest,
        approved: List[OrderDecision],
        rejected: List[OrderDecision],
    ) -> str:
        if request.trade_mode == "single" and (approved or rejected):
            return (approved + rejected)[0].reason
        parts = []
        if approved:
            parts.append(f"{len(approved)} order(s) approved")
        if rejected:
            parts.append(f"{len(rejected)} order(s) rejected")
        if not parts:
            return "No orders to evaluate."
        return "; ".join(parts) + "."

    def _now_str(self) -> str:
        try:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo(self.config.timezone)).strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
        except Exception:  # noqa: BLE001 - bad tz must never crash a trade
            logger.warning("Invalid timezone %r; using UTC.", self.config.timezone)
            return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
