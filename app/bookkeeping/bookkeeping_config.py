"""Environment-driven configuration for the bookkeeping engine.

Every configurable value comes from an environment variable. Missing values
fall back to a safe default, are validated, and produce a clear log warning so
that operators always know what the engine is actually running with.

The module deliberately has *no* dependency on the forecasting code. It only
optionally re-uses the ``GOOGLE_CREDENTIALS`` variable already present in the
project ``.env`` so that a single service-account file can be shared.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:  # pragma: no cover - dotenv is part of the project runtime deps
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - never fatal
    pass

logger = logging.getLogger("bookkeeping.config")

# Repository root: app/bookkeeping/bookkeeping_config.py -> parents[2]
BASE_DIR = Path(__file__).resolve().parents[2]

# The bookkeeping Google Sheet supplied by the project owner. Used only as a
# last-resort default; operators should set BOOKKEEPING_SHEET_ID explicitly.
DEFAULT_SHEET_ID = "1sBWjYXuShw-KLYjOs77EZeYh6lbXIvs4zOE5z_7zCHI"

VALID_BACKENDS = {"google_sheets", "local", "nosql"}
VALID_TRADE_MODES = {"single", "portfolio"}


# --------------------------------------------------------------------------
# Typed environment readers (each one logs a clear warning on bad input)
# --------------------------------------------------------------------------
def _str_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    token = value.strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid bool for %s=%r; using default %s", name, value, default)
    return default


def _int_env(name: str, default: int, *, minimum: int | None = None) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        logger.warning("Invalid int for %s=%r; using default %s", name, value, default)
        return default
    if minimum is not None and parsed < minimum:
        logger.warning(
            "%s=%s below minimum %s; clamping to minimum", name, parsed, minimum
        )
        return minimum
    return parsed


def _float_env(name: str, default: float, *, minimum: float | None = None) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid float for %s=%r; using default %s", name, value, default)
        return default
    if minimum is not None and parsed < minimum:
        logger.warning(
            "%s=%s below minimum %s; clamping to minimum", name, parsed, minimum
        )
        return minimum
    return parsed


@dataclass
class BookkeepingConfig:
    """Resolved, validated configuration for one engine instance."""

    # Storage backend selection.
    backend: str = "google_sheets"

    # Google Sheets settings.
    sheet_id: str = DEFAULT_SHEET_ID
    google_credentials: Path = field(default_factory=lambda: BASE_DIR / "credentials")
    # Worksheet / section names. ``worksheet_name`` is the primary executed-trades
    # tab; ``ledger_sheet_name`` is the full append-only audit ledger.
    worksheet_name: str = "Executed"
    ledger_sheet_name: str = "Ledger"
    rejected_sheet_name: str = "Rejected"
    suggestions_sheet_name: str = "Suggestions"
    state_sheet_name: str = "State"
    request_log_sheet_name: str = "RequestLog"

    # Capital / trading rules.
    initial_capital: float = 100_000.0
    currency: str = "INR"
    trade_mode: str = "single"
    allow_partial_fills: bool = False
    enable_margin: bool = False
    max_symbols_per_batch: int = 20
    # Fraction (0-1] of currently available capital that may be deployed in a
    # single request. 1.0 means "all of it"; 0.8 keeps a 20% cash reserve.
    decision_threshold: float = 1.0

    # Local persistence (also used as a cache / fallback for Sheets backend).
    state_file: Path = field(default_factory=lambda: BASE_DIR / "state" / "bookkeeping_state.json")
    timezone: str = "Asia/Kolkata"

    # Non-fatal warnings accumulated while resolving the configuration.
    warnings: List[str] = field(default_factory=list)

    def as_public_dict(self) -> dict:
        """Config view safe to surface to Claude (no secrets)."""
        return {
            "backend": self.backend,
            "sheet_id": self.sheet_id if self.backend == "google_sheets" else None,
            "worksheet_name": self.worksheet_name,
            "ledger_sheet_name": self.ledger_sheet_name,
            "currency": self.currency,
            "trade_mode": self.trade_mode,
            "allow_partial_fills": self.allow_partial_fills,
            "enable_margin": self.enable_margin,
            "max_symbols_per_batch": self.max_symbols_per_batch,
            "decision_threshold": self.decision_threshold,
            "initial_capital": self.initial_capital,
            "timezone": self.timezone,
        }


def _resolve_credentials() -> Path:
    """Pick a service-account JSON file, preferring the bookkeeping-specific var."""
    candidates = [
        os.getenv("BOOKKEEPING_GOOGLE_CREDENTIALS"),
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),   # primary standard
        os.getenv("GOOGLE_CREDENTIALS"),               # legacy fallback
        str(BASE_DIR / "credentials" / "Credentials_New.json"),
        str(BASE_DIR / "Credentials_New.json"),
        str(BASE_DIR / "Credentials.json"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return Path(candidate).expanduser().resolve()
    # Return the preferred path even if missing so the error surfaces later
    # with a clear message rather than silently doing the wrong thing.
    return Path(candidates[0] or candidates[2]).expanduser()


def load_config() -> BookkeepingConfig:
    """Build a :class:`BookkeepingConfig` from the process environment."""
    warnings: List[str] = []

    backend = _str_env("BOOKKEEPING_BACKEND", "google_sheets").lower()
    if backend not in VALID_BACKENDS:
        warnings.append(
            f"BOOKKEEPING_BACKEND={backend!r} is not one of {sorted(VALID_BACKENDS)}; "
            "falling back to 'google_sheets'."
        )
        backend = "google_sheets"
    if backend == "nosql":
        warnings.append(
            "BOOKKEEPING_BACKEND='nosql' selected but the NoSQL adapter is a "
            "registered stub only; falling back to 'local'."
        )
        backend = "local"

    trade_mode = _str_env("BOOKKEEPING_TRADE_MODE", "single").lower()
    if trade_mode not in VALID_TRADE_MODES:
        warnings.append(
            f"BOOKKEEPING_TRADE_MODE={trade_mode!r} invalid; falling back to 'single'."
        )
        trade_mode = "single"

    decision_threshold = _float_env("BOOKKEEPING_DECISION_THRESHOLD", 1.0)
    if not 0.0 < decision_threshold <= 1.0:
        warnings.append(
            f"BOOKKEEPING_DECISION_THRESHOLD={decision_threshold} outside (0, 1]; "
            "clamping to 1.0."
        )
        decision_threshold = 1.0

    initial_capital = _float_env("BOOKKEEPING_INITIAL_CAPITAL", 100_000.0, minimum=0.0)

    credentials = _resolve_credentials()
    if backend == "google_sheets" and not credentials.exists():
        warnings.append(
            f"Google credentials file not found at {credentials}; the "
            "google_sheets backend will fail until BOOKKEEPING_GOOGLE_CREDENTIALS "
            "points at a valid service-account JSON."
        )

    enable_margin = _bool_env("BOOKKEEPING_ENABLE_MARGIN", False)
    if enable_margin:
        warnings.append(
            "BOOKKEEPING_ENABLE_MARGIN=true requested, but margin logic is not "
            "implemented yet; the engine continues in cash-only mode."
        )
        enable_margin = False

    state_file = Path(
        _str_env("BOOKKEEPING_STATE_FILE", str(BASE_DIR / "state" / "bookkeeping_state.json"))
    ).expanduser()
    if not state_file.is_absolute():
        state_file = (BASE_DIR / state_file).resolve()

    config = BookkeepingConfig(
        backend=backend,
        sheet_id=_str_env("BOOKKEEPING_SHEET_ID", DEFAULT_SHEET_ID),
        google_credentials=credentials,
        worksheet_name=_str_env("BOOKKEEPING_WORKSHEET_NAME", "Executed"),
        ledger_sheet_name=_str_env("BOOKKEEPING_LEDGER_SHEET_NAME", "Ledger"),
        initial_capital=initial_capital,
        currency=_str_env("BOOKKEEPING_CURRENCY", "INR"),
        trade_mode=trade_mode,
        allow_partial_fills=_bool_env("BOOKKEEPING_ALLOW_PARTIAL_FILLS", False),
        enable_margin=enable_margin,
        max_symbols_per_batch=_int_env("BOOKKEEPING_MAX_SYMBOLS_PER_BATCH", 20, minimum=1),
        decision_threshold=decision_threshold,
        state_file=state_file,
        timezone=_str_env("BOOKKEEPING_TIMEZONE", "Asia/Kolkata"),
        warnings=warnings,
    )

    for message in warnings:
        logger.warning(message)

    return config
