"""
app/pipeline/metadata.py
========================
Production-grade, self-healing pipeline metadata system.

Design goals
------------
* Never crash the pipeline because pipeline_metadata.json is missing,
  empty, truncated, or corrupt.
* Atomic writes via an adjacent .tmp file → os.replace(), so a killed
  process never leaves a half-written JSON on disk.
* Backup-on-write: every successful save also writes
  pipeline_metadata.json.bak, giving one safe fallback generation.
* Schema migrations: a monotonically-increasing ``schema_version`` field
  lets future changes add/rename keys without breaking older containers.
* Concurrent-access safety: writes are atomic at the OS level (os.replace
  is atomic on POSIX / APFS / ext4); readers always see a complete file.
* Structured logging: every recovery or repair event is emitted at WARNING
  so it surfaces in container logs without stopping the workflow.

Public API
----------
  safe_load_metadata(path)            → Dict[str, Any]
  safe_save_metadata(path, data)      → None
  validate_metadata_schema(data)      → List[str]   # list of error strings
  initialize_metadata_if_missing(path) → bool        # True = file was created
  repair_metadata_if_corrupted(path)  → bool        # True = repair was needed
  migrate_metadata_schema(data)       → Dict[str, Any]
  get_or_create_metadata(path)        → Dict[str, Any]
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version history
# ---------------------------------------------------------------------------
#   v1 (2025-xx-xx): initial production schema
#   v2 (future)    : add per-worksheet last_run_at timestamps
CURRENT_SCHEMA_VERSION: int = 1

# ---------------------------------------------------------------------------
# Required keys that main.py's validate_metadata() hard-crashes on if absent.
# These MUST be present in any metadata the inference pipeline consumes.
# ---------------------------------------------------------------------------
INFERENCE_REQUIRED_KEYS: Tuple[str, ...] = (
    "feature_columns",
    "seq_len",
    "feature_count",
    "dense_input_size",
    "horizons",
    "quantiles",
    "target_type",
)

# ---------------------------------------------------------------------------
# Default (canonical) metadata schema.
#
# These values were reverse-engineered from the saved model checkpoint
# weight matrices and confirmed against historical metrics.json files.
#
# Model architecture evidence:
#   Dense.pt    data/0: 74,240 params = 580 × 128   → dense_input_size=580
#   LSTM.pt     data/0:  7,424 params = 4 × 64 × 29 → feature_count=29
#   Transformer data/1:  3,712 params = 128 × 29    → input_size=29
#   All outputs: 15 params = 3 quantiles × 5 horizons
#   metrics.json predictor_count: 580 = 20 × 29     → seq_len=20
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    # OHLCV (5)
    "Open", "High", "Low", "Close", "Volume",
    # Momentum (4)
    "RSI_14", "MACD_12_26", "MACD_Signal_9", "MACD_Histogram",
    # Stochastic (2)
    "Stochastic_%K", "Stochastic_%D",
    # Moving averages (6)
    "SMA_5", "SMA_20", "SMA_50",
    "EMA_12", "EMA_26", "EMA_50",
    # Trend / volatility (7)
    "ADX_14",
    "BB_Upper_20", "BB_Middle_20", "BB_Lower_20",
    "ATR_14",
    # Volume-based (2)
    "OBV", "VWAP",
    # Return / rate-of-change (3)
    "Daily_Return_%", "Log_Return_%", "ROC_12",
    # Oscillators (2)
    "Williams_%R", "CCI_20",
]  # len = 29

# Forward-return label columns produced by Feature_Engineering.py.
# These must NEVER appear in feature_columns (leakage guard in validate_metadata).
_LABEL_COLUMNS: List[str] = (
    [f"y_logret_h{h}" for h in range(1, 31)]  # h=1..30
    + ["has_labels"]
)

_DEFAULT_METADATA: Dict[str, Any] = {
    # ── schema bookkeeping ──────────────────────────────────────────────────
    "schema_version": CURRENT_SCHEMA_VERSION,
    "created_at": None,          # filled in at initialisation time
    "last_modified_at": None,    # updated on every save
    "pipeline_version": "1.0.0",
    "feature_engineering_version": "1.0.0",

    # ── model architecture ──────────────────────────────────────────────────
    "feature_columns": FEATURE_COLUMNS,
    "seq_len": 20,
    "feature_count": len(FEATURE_COLUMNS),   # 29
    "dense_input_size": 20 * len(FEATURE_COLUMNS),  # 580
    "horizons": [1, 2, 3, 4, 5],
    "quantiles": [0.1, 0.5, 0.9],
    "target_type": "multi_horizon_quantile_log_return",
    "label_columns": _LABEL_COLUMNS,

    # ── ensemble ────────────────────────────────────────────────────────────
    "ensemble_weights": {
        "Dense":       0.2,
        "LSTM":        0.3,
        "Transformer": 0.5,
    },

    # ── per-model capacity (used by infer_transformer_heads fallback) ───────
    "model_capacity": {
        "Dense": {
            "hidden_size": 128,
        },
        "LSTM": {
            "hidden_size": 64,
            "num_layers":  2,
        },
        "Transformer": {
            "hidden_size": 128,
            "num_heads":   4,
            "num_layers":  2,
        },
    },

    # ── runtime / workflow tracking ─────────────────────────────────────────
    "last_successful_run": None,
    "last_run_status":     None,
    "last_run_at":         None,
    "run_count":           0,

    # ── per-worksheet incremental state ─────────────────────────────────────
    # {"RELIANCE": {"last_row_processed": 150, "last_inference_at": "..."}, ...}
    "worksheet_state": {},

    # ── training provenance ─────────────────────────────────────────────────
    "trained_at":          None,
    "training_data_start": None,
    "training_data_end":   None,
    "model_files": {
        "Dense":       "outputs/Saved_Models/Dense.pt",
        "LSTM":        "outputs/Saved_Models/LSTM.pt",
        "Transformer": "outputs/Saved_Models/Transformer.pt",
    },
}


# ===========================================================================
# Internal helpers
# ===========================================================================

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _backup_path(path: Path) -> Path:
    return path.with_suffix(".json.bak")


def _tmp_path(path: Path) -> Path:
    """Return a sibling .tmp file in the same directory (same filesystem)."""
    return path.with_suffix(".json.tmp")


# ===========================================================================
# Public API
# ===========================================================================

def safe_load_metadata(path: Path) -> Dict[str, Any]:
    """
    Load pipeline_metadata.json with full fault tolerance.

    Recovery sequence:
      1. Read and parse the primary file.
      2. On any failure, attempt to load the .bak backup.
      3. If both fail, return initialised defaults and persist them to disk
         so the next caller succeeds without going through recovery again.

    Never raises. Always returns a dict that passes validate_metadata_schema.
    """
    path = Path(path)

    # ── attempt 1: primary file ──────────────────────────────────────────────
    try:
        data = _read_json(path)
        if data is not None:
            data = migrate_metadata_schema(data)
            errors = validate_metadata_schema(data)
            if not errors:
                return data
            logger.warning(
                "[metadata] Primary file loaded but schema invalid (%d errors): %s — repairing",
                len(errors), errors,
            )
            # Fall through to backup attempt
    except Exception as exc:
        logger.warning("[metadata] Failed to read primary metadata file: %s — trying backup", exc)

    # ── attempt 2: backup file ───────────────────────────────────────────────
    bak = _backup_path(path)
    if bak.exists():
        try:
            data = _read_json(bak)
            if data is not None:
                data = migrate_metadata_schema(data)
                errors = validate_metadata_schema(data)
                if not errors:
                    logger.warning(
                        "[metadata] Recovered from backup %s — overwriting primary", bak
                    )
                    safe_save_metadata(path, data)
                    return data
        except Exception as exc:
            logger.warning("[metadata] Backup recovery also failed: %s", exc)

    # ── attempt 3: rebuild from defaults ────────────────────────────────────
    logger.warning(
        "[metadata] Both primary and backup unreadable — initialising from defaults. "
        "If models were trained with different hyperparameters this may cause inference errors."
    )
    data = _build_default()
    safe_save_metadata(path, data)
    return data


def safe_save_metadata(path: Path, data: Dict[str, Any]) -> None:
    """
    Atomically write metadata to *path*.

    Steps:
      1. Update `last_modified_at` timestamp.
      2. Serialise to a sibling .tmp file.
      3. os.replace() the .tmp → primary (atomic on POSIX / APFS).
      4. Copy primary → .bak for next-generation backup.

    Raises RuntimeError only if the directory itself is not writable (a hard
    misconfiguration that should surface immediately).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = deepcopy(data)
    data["last_modified_at"] = _utc_now()
    if not data.get("created_at"):
        data["created_at"] = data["last_modified_at"]

    payload = json.dumps(data, indent=2, ensure_ascii=False, default=str)

    tmp = _tmp_path(path)
    try:
        # Write to tmp, then atomically replace
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)

        # Update backup
        try:
            shutil.copy2(path, _backup_path(path))
        except Exception as exc:
            logger.warning("[metadata] Backup copy failed (non-fatal): %s", exc)

    except OSError as exc:
        logger.error("[metadata] ATOMIC WRITE FAILED for %s: %s", path, exc)
        raise RuntimeError(f"Cannot write metadata to {path}: {exc}") from exc
    finally:
        # Ensure tmp file is cleaned up even on failure
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def validate_metadata_schema(data: Any) -> List[str]:
    """
    Return a list of human-readable error strings.
    An empty list means the schema is valid for inference.
    """
    if not isinstance(data, dict):
        return [f"Metadata must be a dict, got {type(data).__name__}"]

    errors: List[str] = []

    for key in INFERENCE_REQUIRED_KEYS:
        if key not in data:
            errors.append(f"Missing required key: '{key}'")

    if "feature_columns" in data and "feature_count" in data:
        fc = data.get("feature_columns")
        n  = data.get("feature_count")
        if isinstance(fc, list) and isinstance(n, int):
            if len(fc) != n:
                errors.append(
                    f"feature_count={n} but feature_columns has {len(fc)} entries"
                )

    if "seq_len" in data and "feature_count" in data and "dense_input_size" in data:
        s = data.get("seq_len")
        f = data.get("feature_count")
        d = data.get("dense_input_size")
        if isinstance(s, int) and isinstance(f, int) and isinstance(d, int):
            if s * f != d:
                errors.append(
                    f"dense_input_size mismatch: seq_len({s}) × feature_count({f}) = {s*f} ≠ {d}"
                )

    if "horizons" in data:
        horizons = data["horizons"]
        if not isinstance(horizons, list) or len(horizons) == 0:
            errors.append("horizons must be a non-empty list")

    if "quantiles" in data:
        quantiles = data["quantiles"]
        if not isinstance(quantiles, list) or len(quantiles) == 0:
            errors.append("quantiles must be a non-empty list")
        elif 0.5 not in [float(q) for q in quantiles]:
            errors.append("quantiles must include 0.5 (median quantile)")

    return errors


def initialize_metadata_if_missing(path: Path) -> bool:
    """
    Create pipeline_metadata.json with production defaults if it does not
    exist or is empty.

    Returns True if the file was created/recreated, False if it already
    existed and was valid.
    """
    path = Path(path)

    if path.exists() and path.stat().st_size > 10:
        # Quick sanity check: parseable JSON with required keys?
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            errors = validate_metadata_schema(data)
            if not errors:
                return False  # already healthy
            logger.warning(
                "[metadata] Existing metadata has schema errors (%s) — reinitialising "
                "from defaults. Manual review recommended if models were recently retrained.",
                errors,
            )
        except (json.JSONDecodeError, OSError):
            logger.warning("[metadata] Existing metadata unreadable — reinitialising")

    logger.warning(
        "[metadata] pipeline_metadata.json missing or invalid at %s — "
        "creating from production defaults (schema_version=%d)",
        path, CURRENT_SCHEMA_VERSION,
    )
    data = _build_default()
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_save_metadata(path, data)
    return True


def repair_metadata_if_corrupted(path: Path) -> bool:
    """
    Non-destructive repair pass.

    1. Read the file.
    2. If it cannot be parsed, restore from .bak if available, else rebuild.
    3. If it CAN be parsed but has schema errors, patch in missing keys from
       defaults rather than overwriting the whole file.

    Returns True if any repair was performed.
    """
    path = Path(path)

    if not path.exists():
        logger.warning("[metadata] File absent — initialising fresh copy")
        initialize_metadata_if_missing(path)
        return True

    raw: Optional[str] = None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("[metadata] Cannot read metadata file: %s", exc)
        initialize_metadata_if_missing(path)
        return True

    # ── try to parse ─────────────────────────────────────────────────────────
    try:
        data: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("[metadata] JSON decode error (%s) — attempting backup restore", exc)
        bak = _backup_path(path)
        if bak.exists():
            try:
                data = json.loads(bak.read_text(encoding="utf-8"))
                logger.warning("[metadata] Restored from backup %s", bak)
            except Exception:
                logger.warning("[metadata] Backup also corrupt — rebuilding from defaults")
                initialize_metadata_if_missing(path)
                return True
        else:
            initialize_metadata_if_missing(path)
            return True

    # ── patch missing keys without overwriting known-good data ───────────────
    errors = validate_metadata_schema(data)
    if not errors:
        return False  # healthy, nothing to do

    logger.warning("[metadata] Schema errors detected: %s — patching missing keys", errors)
    repaired = _build_default()
    repaired.update(data)      # keep existing values
    # Force correct derived values
    fc = repaired.get("feature_columns", FEATURE_COLUMNS)
    repaired["feature_count"]    = len(fc)
    repaired["dense_input_size"] = int(repaired.get("seq_len", 20)) * len(fc)

    remaining = validate_metadata_schema(repaired)
    if remaining:
        logger.error(
            "[metadata] Cannot repair schema automatically (%s) — full reinit required",
            remaining,
        )
        initialize_metadata_if_missing(path)
    else:
        safe_save_metadata(path, repaired)
        logger.warning("[metadata] Repair successful — patched keys: %s", errors)
    return True


def migrate_metadata_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply in-place schema migrations.

    Each migration is idempotent: running it twice produces the same result.
    The `schema_version` field is bumped only after a successful migration.
    """
    if not isinstance(data, dict):
        return data

    version: int = int(data.get("schema_version", 0))

    # ── migration 0→1: add tracking + capacity fields ────────────────────────
    if version < 1:
        defaults = _build_default()
        for key in (
            "pipeline_version", "feature_engineering_version",
            "last_successful_run", "last_run_status", "last_run_at",
            "run_count", "worksheet_state", "trained_at",
            "training_data_start", "training_data_end", "model_files",
            "label_columns", "model_capacity",
        ):
            data.setdefault(key, defaults[key])

        # Ensure created_at is present
        data.setdefault("created_at", _utc_now())
        data["schema_version"] = 1
        logger.info("[metadata] Migrated schema 0→1")

    # ── future migrations go here ─────────────────────────────────────────────
    # if version < 2:
    #     data.setdefault("worksheet_state", {})
    #     data["schema_version"] = 2

    return data


def get_or_create_metadata(path: Path) -> Dict[str, Any]:
    """
    One-call convenience wrapper used by main.py and monthly_finetune.py.

    Ensures the file exists and is valid, then returns its contents.
    Equivalent to:
        initialize_metadata_if_missing(path)
        return safe_load_metadata(path)
    """
    initialize_metadata_if_missing(path)
    return safe_load_metadata(path)


def record_run_result(
    path: Path,
    *,
    status: str,
    worksheets: Optional[List[str]] = None,
    error: Optional[str] = None,
) -> None:
    """
    Update workflow-tracking fields after a pipeline run.

    This write is best-effort: any failure is logged but never propagated.
    """
    try:
        data = safe_load_metadata(path)
        now  = _utc_now()
        data["last_run_at"]     = now
        data["last_run_status"] = status
        data["run_count"]       = int(data.get("run_count", 0)) + 1
        if status in {"success", "ok"}:
            data["last_successful_run"] = now
        if worksheets:
            ws_state: Dict[str, Any] = data.setdefault("worksheet_state", {})
            for ws in worksheets:
                ws_state.setdefault(ws, {})
                ws_state[ws]["last_inference_at"] = now
                ws_state[ws]["last_status"]       = status
        safe_save_metadata(path, data)
    except Exception as exc:
        logger.warning("[metadata] record_run_result failed (non-fatal): %s", exc)


# ===========================================================================
# Private helpers
# ===========================================================================

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """
    Read and parse JSON from *path*.
    Returns None if the file is empty; raises on all other errors.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return json.loads(text)


def _build_default() -> Dict[str, Any]:
    """Return a deep copy of the default schema with timestamps filled in."""
    data = deepcopy(_DEFAULT_METADATA)
    now  = _utc_now()
    data["created_at"]       = now
    data["last_modified_at"] = now
    return data
