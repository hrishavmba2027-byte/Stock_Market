"""Storage abstraction for the bookkeeping engine.

The :class:`StorageAdapter` interface is the single seam between the decision
logic and *where* data lives. The engine never imports a concrete backend; it
only ever talks to this interface. Adding a new backend (e.g. a NoSQL store)
therefore means writing one new adapter and registering it in
:func:`get_storage` -- no decision logic changes.

Backends shipped here
---------------------
* ``local``         -- :class:`LocalJSONStorage`, a JSON-file ledger. Always
                       available; used for tests, offline use, and as the
                       cache layer for the Sheets backend.
* ``google_sheets`` -- :class:`~app.bookkeeping.bookkeeping_google_sheets.GoogleSheetsStorage`
                       (imported lazily so this module has no hard gspread dep).
* ``nosql``         -- :class:`NoSqlStoragePlaceholder`, a documented stub that
                       shows exactly where a Mongo/DynamoDB/Firestore adapter
                       would slot in.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.bookkeeping.bookkeeping_config import BookkeepingConfig
from app.bookkeeping.bookkeeping_models import BookkeepingState, LedgerRow

logger = logging.getLogger("bookkeeping.storage")

# Logical tables and their column order. Every backend must honour this schema
# so that the logical model is identical regardless of the physical store.
TABLE_COLUMNS: Dict[str, List[str]] = {
    "ledger": LedgerRow.COLUMNS,
    "executed": LedgerRow.COLUMNS,
    "rejected": LedgerRow.COLUMNS,
    "suggestions": [
        "Date", "Request_ID", "Symbol", "Side", "Qty", "Price",
        "Order_Value", "Trade_Mode", "Note",
    ],
    "request_log": [
        "Date", "Request_ID", "Trade_Mode", "Dry_Run", "Can_Trade",
        "Request_JSON", "Response_JSON",
    ],
}


class StorageAdapter(ABC):
    """Backend-agnostic persistence contract used by the engine."""

    @abstractmethod
    def init_schema(self) -> None:
        """Create any tables/worksheets/headers that do not yet exist."""

    @abstractmethod
    def load_state(self) -> BookkeepingState:
        """Return the latest persisted :class:`BookkeepingState`."""

    @abstractmethod
    def save_state(self, state: BookkeepingState) -> None:
        """Persist the supplied :class:`BookkeepingState`."""

    @abstractmethod
    def append_records(self, table: str, records: List[Dict[str, Any]]) -> None:
        """Append rows to ``table`` (one of :data:`TABLE_COLUMNS`)."""

    @abstractmethod
    def health_check(self) -> Tuple[bool, str]:
        """Return ``(ok, human_readable_detail)`` for diagnostics."""

    # Optional helper -- backends may override for richer reporting.
    def describe(self) -> str:
        return self.__class__.__name__


# --------------------------------------------------------------------------
# Local JSON backend
# --------------------------------------------------------------------------
class LocalJSONStorage(StorageAdapter):
    """A durable, append-only ledger backed by plain JSON / JSONL files.

    State lives in ``config.state_file``; each table is a sibling ``.jsonl``
    file. Writes are atomic (temp file + ``os.replace``) so a crash mid-write
    never corrupts the ledger.
    """

    def __init__(self, config: BookkeepingConfig) -> None:
        self.config = config
        self.state_path = Path(config.state_file)
        self.dir = self.state_path.parent
        self.dir.mkdir(parents=True, exist_ok=True)

    def _table_path(self, table: str) -> Path:
        return self.dir / f"bookkeeping_{table}.jsonl"

    def init_schema(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for table in TABLE_COLUMNS:
            path = self._table_path(table)
            if not path.exists():
                path.touch()
        if not self.state_path.exists():
            self.save_state(
                BookkeepingState(
                    initial_capital=self.config.initial_capital,
                    available_capital=self.config.initial_capital,
                    last_updated=_now_iso(),
                )
            )
        logger.info("Local storage initialised at %s", self.dir)

    def load_state(self) -> BookkeepingState:
        if not self.state_path.exists():
            return BookkeepingState(
                initial_capital=self.config.initial_capital,
                available_capital=self.config.initial_capital,
                last_updated=_now_iso(),
            )
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8") or "{}")
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Bookkeeping state file is unreadable: {exc}") from exc
        return BookkeepingState.from_dict(raw)

    def save_state(self, state: BookkeepingState) -> None:
        state.last_updated = _now_iso()
        _atomic_write(self.state_path, json.dumps(state.to_dict(), indent=2))

    def append_records(self, table: str, records: List[Dict[str, Any]]) -> None:
        if table not in TABLE_COLUMNS:
            raise ValueError(f"Unknown table {table!r}.")
        if not records:
            return
        path = self._table_path(table)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, default=str) + "\n")

    def health_check(self) -> Tuple[bool, str]:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            probe = self.dir / ".bk_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True, f"local backend writable at {self.dir}"
        except OSError as exc:
            return False, f"local backend not writable: {exc}"

    def describe(self) -> str:
        return f"LocalJSONStorage(dir={self.dir})"


# --------------------------------------------------------------------------
# NoSQL placeholder -- documents the extension point
# --------------------------------------------------------------------------
class NoSqlStoragePlaceholder(StorageAdapter):
    """Registered but inert NoSQL adapter.

    A real implementation (MongoDB, DynamoDB, Firestore, ...) only needs to
    implement the five :class:`StorageAdapter` methods. Because the engine
    depends solely on that interface, swapping it in requires *zero* changes to
    the decision logic -- you would map:

    * ``load_state`` / ``save_state``  -> a single ``state`` document
    * ``append_records``               -> ``insert_many`` into a per-table collection
    * ``init_schema``                  -> create collections + indexes (idempotent)

    Until then this class fails loudly so misconfiguration is never silent.
    """

    def __init__(self, config: BookkeepingConfig) -> None:
        self.config = config

    def _unavailable(self) -> RuntimeError:
        return RuntimeError(
            "NoSQL backend is not implemented. Implement a StorageAdapter "
            "subclass and register it in get_storage(); the engine needs no "
            "other changes."
        )

    def init_schema(self) -> None:
        raise self._unavailable()

    def load_state(self) -> BookkeepingState:
        raise self._unavailable()

    def save_state(self, state: BookkeepingState) -> None:
        raise self._unavailable()

    def append_records(self, table: str, records: List[Dict[str, Any]]) -> None:
        raise self._unavailable()

    def health_check(self) -> Tuple[bool, str]:
        return False, "nosql backend is a stub; see NoSqlStoragePlaceholder docstring"


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def get_storage(config: BookkeepingConfig) -> StorageAdapter:
    """Return the :class:`StorageAdapter` selected by ``config.backend``.

    This is the *only* place that knows about concrete backends.
    """
    backend = config.backend
    if backend == "local":
        return LocalJSONStorage(config)
    if backend == "nosql":
        return NoSqlStoragePlaceholder(config)
    if backend == "google_sheets":
        # Lazy import keeps gspread an optional dependency for local/test use.
        from app.bookkeeping.bookkeeping_google_sheets import GoogleSheetsStorage

        return GoogleSheetsStorage(config)
    raise ValueError(f"Unsupported BOOKKEEPING_BACKEND={backend!r}.")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
