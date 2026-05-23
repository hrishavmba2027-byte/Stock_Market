"""
app/pipeline/startup.py
=======================
Startup self-validation system for the Stock Market ML pipeline.

Called once at process start (from app_data.py and the watcher entrypoint)
to detect misconfiguration early — before any network calls or model loads.

Checks performed
----------------
  1. Required runtime directories exist and are writable.
  2. Saved model checkpoints are present (Dense.pt, LSTM.pt, Transformer.pt).
  3. Google credentials file is present and is a non-empty JSON.
  4. pipeline_metadata.json exists and passes schema validation; created from
     defaults if absent.
  5. Key environment variables are set (warns; does not abort).
  6. Filesystem write-test on outputs/ to catch Docker read-only mount issues.

Severity levels
---------------
  CRITICAL  — system cannot function at all; raises RuntimeError immediately.
  WARNING   — service can start but a feature is degraded; logs only.
  INFO      — informational status confirmation.

Usage
-----
  from app.pipeline.startup import run_startup_checks
  run_startup_checks(settings)     # call once at process start
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.pipeline.metadata import (
    get_or_create_metadata,
    initialize_metadata_if_missing,
    validate_metadata_schema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model files that MUST exist for inference to function
# ---------------------------------------------------------------------------
REQUIRED_MODEL_FILES: Tuple[str, ...] = ("Dense.pt", "LSTM.pt", "Transformer.pt")

# ---------------------------------------------------------------------------
# Directories that must be writable at runtime
# ---------------------------------------------------------------------------
REQUIRED_WRITABLE_DIRS: Tuple[str, ...] = (
    "logs",
    "outputs",
    "outputs/main_inference",
    "state",
)

# ---------------------------------------------------------------------------
# Environment variables that must be set for full functionality
# (WARNING level — missing values degrade features, don't abort startup)
# ---------------------------------------------------------------------------
RECOMMENDED_ENV_VARS: Tuple[str, ...] = (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "SHEET_ID",
    "DEVICE",
)


# ===========================================================================
# Main entry point
# ===========================================================================

def run_startup_checks(settings: Any, *, strict: bool = False) -> None:
    """
    Run all startup self-checks.

    Parameters
    ----------
    settings : app.config.settings.Settings
        The application settings object (provides paths, credentials, etc.).
    strict : bool
        If True, any WARNING-level issue also raises RuntimeError.
        Useful for CI / pre-deploy validation.

    Raises
    ------
    RuntimeError
        If any CRITICAL check fails, or if strict=True and any WARNING fires.
    """
    base = Path(settings.base_dir)
    issues: List[Tuple[str, str]] = []   # (severity, message)

    _check_writable_dirs(base, issues)
    _check_model_files(Path(settings.model_dir), issues)
    _check_credentials(Path(settings.google_credentials), issues)
    _check_metadata(Path(settings.metadata_path), issues)
    _check_env_vars(issues)
    _check_write_test(Path(settings.outputs_dir), issues)

    # ── report ───────────────────────────────────────────────────────────────
    criticals = [(s, m) for s, m in issues if s == "CRITICAL"]
    warnings  = [(s, m) for s, m in issues if s == "WARNING"]
    infos     = [(s, m) for s, m in issues if s == "INFO"]

    for _, msg in infos:
        logger.info("[startup] ✓ %s", msg)
    for _, msg in warnings:
        logger.warning("[startup] ⚠ %s", msg)
    for _, msg in criticals:
        logger.critical("[startup] ✗ CRITICAL: %s", msg)

    if criticals:
        summary = "; ".join(m for _, m in criticals)
        raise RuntimeError(f"Startup validation failed — {len(criticals)} critical issues: {summary}")

    if strict and warnings:
        summary = "; ".join(m for _, m in warnings)
        raise RuntimeError(f"Strict startup validation failed — {len(warnings)} warnings: {summary}")

    if not issues:
        logger.info("[startup] All startup checks passed ✓")
    else:
        logger.info(
            "[startup] Startup checks complete — %d critical, %d warnings, %d info",
            len(criticals), len(warnings), len(infos),
        )


# ===========================================================================
# Individual checks
# ===========================================================================

def _check_writable_dirs(base: Path, issues: List[Tuple[str, str]]) -> None:
    """Ensure all runtime directories exist and are writable."""
    for rel in REQUIRED_WRITABLE_DIRS:
        dirpath = base / rel
        try:
            dirpath.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            issues.append(("CRITICAL", f"Cannot create directory {dirpath}: {exc}"))
            continue

        if not os.access(dirpath, os.W_OK):
            issues.append(("CRITICAL", f"Directory is not writable: {dirpath}"))
        else:
            issues.append(("INFO", f"Directory OK: {rel}/"))


def _check_model_files(model_dir: Path, issues: List[Tuple[str, str]]) -> None:
    """Verify that all saved model checkpoints are present."""
    if not model_dir.exists():
        issues.append((
            "CRITICAL",
            f"Model directory does not exist: {model_dir}. "
            "Inference is impossible without trained checkpoints.",
        ))
        return

    for fname in REQUIRED_MODEL_FILES:
        fpath = model_dir / fname
        if not fpath.exists():
            issues.append((
                "CRITICAL",
                f"Missing model checkpoint: {fpath}. "
                "Run training or restore from a backup release.",
            ))
        elif fpath.stat().st_size < 1024:
            issues.append((
                "WARNING",
                f"Model checkpoint is suspiciously small ({fpath.stat().st_size} bytes): {fpath}",
            ))
        else:
            issues.append(("INFO", f"Model OK: {fname} ({fpath.stat().st_size:,} bytes)"))


def _check_credentials(cred_path: Path, issues: List[Tuple[str, str]]) -> None:
    """Verify Google service-account credentials are present and valid JSON."""
    if not cred_path.exists():
        issues.append((
            "CRITICAL",
            f"Google credentials file not found: {cred_path}. "
            "Set GOOGLE_APPLICATION_CREDENTIALS to the correct path.",
        ))
        return

    try:
        data = json.loads(cred_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        issues.append(("CRITICAL", f"Credentials file is not valid JSON: {cred_path}: {exc}"))
        return

    if "type" not in data or "project_id" not in data:
        issues.append((
            "WARNING",
            f"Credentials file at {cred_path} may not be a service-account key "
            "(missing 'type' or 'project_id'). Authentication may fail.",
        ))
    else:
        issues.append(("INFO", f"Credentials OK: {cred_path.name} (project={data.get('project_id')})"))


def _check_metadata(metadata_path: Path, issues: List[Tuple[str, str]]) -> None:
    """Ensure pipeline_metadata.json exists and passes schema validation."""
    created = initialize_metadata_if_missing(metadata_path)
    if created:
        issues.append((
            "WARNING",
            f"pipeline_metadata.json was absent — created from defaults at {metadata_path}. "
            "If models were retrained with different hyperparameters this default may be stale.",
        ))

    try:
        text = metadata_path.read_text(encoding="utf-8").strip()
        data = json.loads(text) if text else {}
    except Exception as exc:
        issues.append((
            "CRITICAL",
            f"Cannot read pipeline_metadata.json after initialisation: {exc}",
        ))
        return

    errors = validate_metadata_schema(data)
    if errors:
        issues.append((
            "CRITICAL",
            f"pipeline_metadata.json schema invalid after repair attempt: {errors}",
        ))
    else:
        fc   = len(data.get("feature_columns", []))
        s    = data.get("seq_len", "?")
        h    = data.get("horizons", [])
        issues.append((
            "INFO",
            f"Metadata OK: {fc} features, seq_len={s}, horizons={h}",
        ))


def _check_env_vars(issues: List[Tuple[str, str]]) -> None:
    """Warn about missing recommended environment variables."""
    for var in RECOMMENDED_ENV_VARS:
        val = os.environ.get(var)
        if not val:
            issues.append(("WARNING", f"Environment variable not set: {var}"))
        else:
            # Truncate long values (credentials paths etc.) for log safety
            display = val if len(val) < 60 else val[:57] + "..."
            issues.append(("INFO", f"Env {var}={display}"))


def _check_write_test(outputs_dir: Path, issues: List[Tuple[str, str]]) -> None:
    """Perform an actual write test inside outputs/ to catch read-only mounts."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    probe = outputs_dir / ".write_probe"
    try:
        probe.write_text(str(time.time()), encoding="utf-8")
        probe.unlink()
        issues.append(("INFO", "Filesystem write-test passed on outputs/"))
    except OSError as exc:
        issues.append((
            "CRITICAL",
            f"outputs/ directory is NOT writable: {exc}. "
            "Check Docker volume mount permissions.",
        ))
