"""Google Sheets storage adapter for the bookkeeping engine.

This implements :class:`~app.bookkeeping.bookkeeping_storage.StorageAdapter`
against a Google Sheet. It is the default production backend.

Worksheet layout (all created on demand by :meth:`init_schema`)
---------------------------------------------------------------
* ``Ledger``      -- append-only audit row for *every* evaluated order.
* ``Executed``    -- approved + executed orders only.
* ``Rejected``    -- rejected orders only.
* ``Suggestions`` -- Claude's open / dry-run suggestions.
* ``RequestLog``  -- raw request/response JSON for full traceability.
* ``State``       -- a key/value sheet holding the single source of truth for
                     initial capital, available capital, realized & cumulative
                     P&L, open FIFO lots and the processed-request-id set.

``gspread`` is imported lazily and only here, so the rest of the package has no
hard dependency on it. ``gspread>=6`` and ``google-auth`` are already in the
project ``requirements.txt``.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Tuple, TypeVar

from app.bookkeeping.bookkeeping_config import BookkeepingConfig
from app.bookkeeping.bookkeeping_models import BookkeepingState
from app.bookkeeping.bookkeeping_storage import StorageAdapter, TABLE_COLUMNS

logger = logging.getLogger("bookkeeping.google_sheets")

T = TypeVar("T")

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

STATE_HEADERS = ["Key", "Value"]


class GoogleSheetsStorageError(RuntimeError):
    """Raised when the Sheets backend cannot complete an operation."""


class GoogleSheetsStorage(StorageAdapter):
    """Persist the bookkeeping ledger in a Google Sheet."""

    def __init__(self, config: BookkeepingConfig, *, retries: int = 4,
                 backoff_seconds: float = 1.5) -> None:
        self.config = config
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self._client = None
        self._spreadsheet = None
        # Logical table -> physical worksheet name.
        self._table_titles: Dict[str, str] = {
            "ledger": config.ledger_sheet_name,
            "executed": config.worksheet_name,
            "rejected": config.rejected_sheet_name,
            "suggestions": config.suggestions_sheet_name,
            "request_log": config.request_log_sheet_name,
        }

    # -- connection -------------------------------------------------------
    def _gspread(self):
        try:
            import gspread  # noqa: WPS433 (intentional lazy import)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise GoogleSheetsStorageError(
                "The 'gspread' package is required for BOOKKEEPING_BACKEND="
                "google_sheets. Install it with `pip install gspread google-auth` "
                "or switch to BOOKKEEPING_BACKEND=local."
            ) from exc
        return gspread

    def client(self):
        if self._client is None:
            credentials = self.config.google_credentials
            if not credentials.exists():
                raise GoogleSheetsStorageError(
                    f"Google credentials not found at {credentials}. Set "
                    "BOOKKEEPING_GOOGLE_CREDENTIALS to a valid service-account JSON."
                )
            gspread = self._gspread()
            try:
                self._client = gspread.service_account(
                    filename=str(credentials), scopes=GOOGLE_SCOPES
                )
            except TypeError:
                # Older gspread signatures do not accept `scopes`.
                self._client = gspread.service_account(filename=str(credentials))
            logger.info("Authorised Google Sheets service account.")
        return self._client

    def spreadsheet(self):
        if self._spreadsheet is None:
            self._spreadsheet = self._retry(
                lambda: self.client().open_by_key(self.config.sheet_id),
                "open_spreadsheet",
            )
        return self._spreadsheet

    def _retry(self, func: Callable[[], T], operation: str) -> T:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001 - retried & re-raised below
                last_error = exc
                logger.warning(
                    "Sheets op '%s' failed (attempt %s/%s): %s",
                    operation, attempt, self.retries, exc,
                )
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds * attempt)
        raise GoogleSheetsStorageError(
            f"Google Sheets operation '{operation}' failed after {self.retries} "
            f"attempts: {last_error}"
        ) from last_error

    # -- worksheet helpers ------------------------------------------------
    def _get_or_create_worksheet(self, title: str, headers: List[str]):
        spreadsheet = self.spreadsheet()
        try:
            worksheet = self._retry(
                lambda: spreadsheet.worksheet(title), f"worksheet:{title}"
            )
        except GoogleSheetsStorageError:
            worksheet = self._retry(
                lambda: spreadsheet.add_worksheet(
                    title=title, rows=200, cols=max(len(headers), 4)
                ),
                f"add_worksheet:{title}",
            )
            logger.info("Created worksheet %r", title)
        # Ensure the header row exists / is correct.
        existing = self._retry(
            lambda: worksheet.row_values(1), f"row_values:{title}"
        )
        if existing[: len(headers)] != headers:
            self._retry(
                lambda: worksheet.update(
                    range_name="A1", values=[headers]
                ),
                f"write_headers:{title}",
            )
        return worksheet

    # -- StorageAdapter interface ----------------------------------------
    def init_schema(self) -> None:
        for table, title in self._table_titles.items():
            self._get_or_create_worksheet(title, TABLE_COLUMNS[table])
        self._get_or_create_worksheet(self.config.state_sheet_name, STATE_HEADERS)
        # Seed an initial state row if the State sheet is empty.
        state_ws = self._get_or_create_worksheet(
            self.config.state_sheet_name, STATE_HEADERS
        )
        values = self._retry(state_ws.get_all_values, "get_all_values:State")
        if len(values) <= 1:
            self.save_state(
                BookkeepingState(
                    initial_capital=self.config.initial_capital,
                    available_capital=self.config.initial_capital,
                    last_updated=_now_iso(),
                )
            )
        logger.info("Google Sheets schema initialised for sheet %s", self.config.sheet_id)

    def load_state(self) -> BookkeepingState:
        worksheet = self._get_or_create_worksheet(
            self.config.state_sheet_name, STATE_HEADERS
        )
        rows = self._retry(worksheet.get_all_values, "get_all_values:State")
        kv: Dict[str, str] = {}
        for row in rows[1:]:
            if len(row) >= 2 and row[0]:
                kv[row[0].strip()] = row[1]
        if not kv:
            return BookkeepingState(
                initial_capital=self.config.initial_capital,
                available_capital=self.config.initial_capital,
                last_updated=_now_iso(),
            )

        def _num(key: str, default: float = 0.0) -> float:
            try:
                return float(kv.get(key, default) or default)
            except (TypeError, ValueError):
                return default

        def _json(key: str, default):
            try:
                return json.loads(kv.get(key) or json.dumps(default))
            except (TypeError, ValueError, json.JSONDecodeError):
                return default

        return BookkeepingState(
            initial_capital=_num("Initial_Capital", self.config.initial_capital),
            available_capital=_num("Available_Capital", self.config.initial_capital),
            realized_pnl=_num("Realized_PnL"),
            cumulative_pnl=_num("Cumulative_PnL"),
            open_lots=_json("Open_Lots", {}),
            processed_request_ids=_json("Processed_Request_IDs", []),
            last_updated=kv.get("Last_Updated", ""),
            trade_count=int(_num("Trade_Count")),
        )

    def save_state(self, state: BookkeepingState) -> None:
        state.last_updated = _now_iso()
        worksheet = self._get_or_create_worksheet(
            self.config.state_sheet_name, STATE_HEADERS
        )
        rows = [
            STATE_HEADERS,
            ["Initial_Capital", state.initial_capital],
            ["Available_Capital", state.available_capital],
            ["Realized_PnL", state.realized_pnl],
            ["Cumulative_PnL", state.cumulative_pnl],
            ["Trade_Count", state.trade_count],
            ["Last_Updated", state.last_updated],
            ["Open_Lots", json.dumps(state.open_lots, default=str)],
            ["Processed_Request_IDs", json.dumps(state.processed_request_ids)],
        ]
        # Clear then write so stale keys never linger.
        self._retry(worksheet.clear, "clear:State")
        self._retry(
            lambda: worksheet.update(range_name="A1", values=rows),
            "write:State",
        )

    def append_records(self, table: str, records: List[Dict[str, Any]]) -> None:
        if table not in TABLE_COLUMNS:
            raise ValueError(f"Unknown table {table!r}.")
        if not records:
            return
        title = self._table_titles[table]
        headers = TABLE_COLUMNS[table]
        worksheet = self._get_or_create_worksheet(title, headers)
        rows = [[record.get(col, "") for col in headers] for record in records]
        self._retry(
            lambda: worksheet.append_rows(rows, value_input_option="USER_ENTERED"),
            f"append_rows:{title}",
        )

    def health_check(self) -> Tuple[bool, str]:
        try:
            spreadsheet = self.spreadsheet()
            title = self._retry(lambda: spreadsheet.title, "read_title")
            return True, f"connected to Google Sheet '{title}' ({self.config.sheet_id})"
        except Exception as exc:  # noqa: BLE001
            return False, f"google_sheets backend unavailable: {exc}"

    def describe(self) -> str:
        return f"GoogleSheetsStorage(sheet_id={self.config.sheet_id})"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
