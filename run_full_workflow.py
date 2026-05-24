#!/usr/bin/env python3
# =============================================================================
# run_full_workflow.py
# -----------------------------------------------------------------------------
# Master end-to-end orchestrator for the Stock Market ML automation pipeline.
#
# This script is the single entry point that activates the ENTIRE pipeline in
# the correct, deterministic sequence. It does NOT re-implement feature
# engineering, model inference, Google Sheets I/O or the train/test rollover —
# those already exist and are battle-tested inside:
#
#     Data_update.py              — yfinance ingestion + incremental sheet append
#     Feature_Engineering.py      — compute_indicators() (29 indicators + labels)
#     main.py  (run_pipeline)     — per-symbol FE + ensemble inference +
#                                   forecast push + train/test rollover
#     app/pipeline/metadata.py    — self-healing metadata
#     app/pipeline/startup.py     — startup self-validation
#     app/services/sheet_archival.py — TEST -> TRAIN rollover (keep latest 30)
#
# The orchestrator DRIVES those components in 15 explicit stages, inserts
# hard validation gates between them, and adds structured logging, retry /
# exponential backoff, atomic writes, concurrency safety and failure isolation.
#
# 15-stage contract (strict order):
#    1  Startup validation
#    2  Check YFinance for new data
#    3  Download / append new market data
#    4  Update local workbook / data store
#    5  Sync / append new rows into Google Sheets
#    6  Validate workbook / sheet integrity
#    7  Run Feature Engineering
#    8  Validate engineered features
#    9  Update metadata
#   10  Run model pipeline using main.py
#   11  Generate forecasts / predictions
#   12  Push forecast results back into Google Sheets
#   13  Handle train/test sheet rollover logic
#   14  Validate final outputs
#   15  Log workflow summary
#
# Safety model:
#   * Dry-run is the DEFAULT. No Google Sheet is mutated unless --live is passed.
#   * Hard gates (stages 1, 2, 6, 7, 8, 10): a failure aborts the run; later
#     stages are NOT executed. The orchestrator never continues past a failed
#     feature-engineering or model-inference stage.
#   * Every failure is logged loudly (failures log + structured log + stderr).
#     The workflow never silently fails.
#
# Usage:
#   python run_full_workflow.py                      # dry-run, all worksheets
#   python run_full_workflow.py --live               # real end-to-end run
#   python run_full_workflow.py --worksheets RELIANCE,TCS
#   python run_full_workflow.py --live --worksheets RELIANCE
#
# Exit codes:  0 success/skipped   1 failure
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Path bootstrap — make the repo importable regardless of the current working
# directory (Docker /app, host repo root, or a cron invocation elsewhere).
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(os.environ.get("BASE_DIR") or Path(__file__).resolve().parent)
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in {str(PROJECT_ROOT), str(_SCRIPT_DIR)}:
    if _p not in sys.path:
        sys.path.insert(0, _p)

WORKFLOW_VERSION = "1.0.0"
STAGE_COUNT = 15

# Worksheets in the operational sheet that are NOT per-stock data tabs and must
# never be treated as stocks by the orchestrator.
NON_STOCK_TABS = {"SUMMARY", "META", "METADATA", "CONFIG", "DASHBOARD", "INDEX", "SHEET1"}

# Canonical OHLCV schema every stock worksheet must expose (after normalisation).
REQUIRED_SHEET_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Rolling window kept in the TEST (operational) sheet — must match
# app/services/sheet_archival.DEFAULT_ROLLING_OPERATIONAL_ROWS and
# main.LATEST_SHEET_ROWS_TO_KEEP.
ROLLING_TEST_ROWS = int(os.environ.get("ROLLING_OPERATIONAL_ROWS", "30"))

# Default spreadsheet IDs (overridable via env). These match the IDs the user
# supplied and the constants baked into the repository.
DEFAULT_TEST_SHEET_ID = "1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o"
DEFAULT_TRAIN_SHEET_ID = "1_gpRclY80tH3r54b9R5HTXqKF7R7bnMoWugF9Fy-boI"

# Hard gate stages — a failure here aborts the remaining workflow.
GATING_STAGES = {1, 2, 6, 7, 8, 10}


# =========================================================================== #
# Exceptions
# =========================================================================== #
class WorkflowError(RuntimeError):
    """Base class for all orchestrator-level failures."""


class StageFailure(WorkflowError):
    """Raised inside a stage to signal a controlled, logged failure."""


# =========================================================================== #
# Small generic helpers
# =========================================================================== #
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def _run_id() -> str:
    return f"wf-{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (tmp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(Path(path), json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def is_transient_error(exc: BaseException) -> bool:
    """Heuristic: should this error be retried? Quota / network / 5xx → yes."""
    text = f"{type(exc).__name__}: {exc}".lower()
    transient_markers = (
        "quota", "rate", "429", "500", "502", "503", "504",
        "timeout", "timed out", "temporarily", "unavailable",
        "connection", "broken pipe", "reset by peer", "ssl",
        "deadline", "internal error", "backenderror",
    )
    return any(marker in text for marker in transient_markers)


def retry_with_backoff(
    fn: Callable[[], Any],
    *,
    label: str,
    attempts: int,
    base_delay: float,
    max_delay: float = 60.0,
    logger: Optional["WorkflowLogger"] = None,
    component: str = "retry",
    only_transient: bool = True,
) -> Any:
    """
    Execute *fn* with bounded exponential backoff + jitter.

    Retries are attempted only for transient errors when *only_transient* is
    True; deterministic errors fail fast so the workflow never masks a bug.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - deliberate broad guard
            last_exc = exc
            retriable = (not only_transient) or is_transient_error(exc)
            if attempt >= attempts or not retriable:
                if logger:
                    logger.event(
                        component, "ERROR", "retry_exhausted",
                        f"{label}: giving up after {attempt} attempt(s)",
                        error=str(exc), retriable=retriable,
                    )
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)  # jitter
            if logger:
                logger.event(
                    component, "WARNING", "retry",
                    f"{label}: attempt {attempt}/{attempts} failed — "
                    f"retrying in {delay:.1f}s",
                    error=str(exc),
                )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


@contextmanager
def file_lock(lock_path: Path):
    """
    Cross-process exclusive lock so two workflow runs never overlap
    (concurrency safety). Falls back to a no-op if fcntl is unavailable.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w", encoding="utf-8")
    locked = False
    try:
        try:
            import fcntl  # POSIX only
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except ImportError:
            locked = True  # non-POSIX: best-effort, proceed
        except OSError as exc:
            raise WorkflowError(
                f"Another workflow run holds the lock {lock_path}: {exc}. "
                "Refusing to start a concurrent run."
            ) from exc
        handle.write(f"{os.getpid()} {utc_iso()}\n")
        handle.flush()
        yield
    finally:
        try:
            if locked:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        handle.close()
        try:
            lock_path.unlink()
        except OSError:
            pass


# =========================================================================== #
# Configuration
# =========================================================================== #
@dataclass
class WorkflowConfig:
    """Resolved, Docker-safe runtime configuration."""
    base_dir: Path
    credentials_path: Path
    test_sheet_id: str
    train_sheet_id: str
    logs_dir: Path
    workflow_logs_dir: Path
    outputs_dir: Path
    output_dir: Path            # outputs/main_inference
    model_dir: Path             # outputs/Saved_Models  (PROTECTED — never written)
    metadata_path: Path
    state_dir: Path
    workflow_dir: Path          # outputs/workflow
    data_store_dir: Path        # outputs/workflow/data_store
    engineered_dir: Path        # outputs/workflow/engineered
    update_start_date: str
    update_interval: str
    device: str
    subprocess_timeout: int
    subprocess_retries: int
    google_retries: int
    retry_backoff: float

    @staticmethod
    def resolve(base_dir: Path, cli_credentials: Optional[str]) -> "WorkflowConfig":
        base = Path(base_dir).resolve()

        def _resolve(p: str) -> Path:
            pp = Path(p).expanduser()
            return pp if pp.is_absolute() else (base / pp).resolve()

        cred_candidates = [
            cli_credentials,
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
            os.environ.get("GOOGLE_CREDENTIALS"),
            "/app/credentials/Credentials_New.json",          # Docker
            str(base / "credentials" / "Credentials_New.json"),  # host
        ]
        credentials_path = next(
            (_resolve(c) for c in cred_candidates if c and _resolve(c).exists()),
            _resolve(str(base / "credentials" / "Credentials_New.json")),
        )

        outputs_dir = _resolve(os.environ.get("OUTPUTS_DIR", "outputs"))
        logs_dir = _resolve(os.environ.get("LOGS_DIR", "logs"))
        return WorkflowConfig(
            base_dir=base,
            credentials_path=credentials_path,
            test_sheet_id=os.environ.get("SHEET_ID")
            or os.environ.get("OPERATIONAL_SHEET_ID")
            or DEFAULT_TEST_SHEET_ID,
            train_sheet_id=os.environ.get("HISTORICAL_TRAINING_SHEET_ID")
            or os.environ.get("TRAINING_SHEET_ID")
            or DEFAULT_TRAIN_SHEET_ID,
            logs_dir=logs_dir,
            workflow_logs_dir=logs_dir / "workflow",
            outputs_dir=outputs_dir,
            output_dir=_resolve(os.environ.get("OUTPUT_DIR", str(outputs_dir / "main_inference"))),
            model_dir=_resolve(os.environ.get("MODEL_DIR", str(outputs_dir / "Saved_Models"))),
            metadata_path=_resolve(os.environ.get("METADATA_PATH", str(outputs_dir / "pipeline_metadata.json"))),
            state_dir=_resolve(os.environ.get("STATE_DIR", "state")),
            workflow_dir=outputs_dir / "workflow",
            data_store_dir=outputs_dir / "workflow" / "data_store",
            engineered_dir=outputs_dir / "workflow" / "engineered",
            update_start_date=os.environ.get("UPDATE_START_DATE", "2015-01-01"),
            update_interval=os.environ.get("UPDATE_INTERVAL", "1d"),
            device=os.environ.get("DEVICE", "auto"),
            subprocess_timeout=int(os.environ.get("SUBPROCESS_TIMEOUT_SECONDS", "1800")),
            subprocess_retries=int(os.environ.get("SUBPROCESS_RETRIES", "2")),
            google_retries=int(os.environ.get("GOOGLE_RETRIES", "3")),
            retry_backoff=float(os.environ.get("RETRY_BACKOFF_SECONDS", "2.0")),
        )

    def ensure_dirs(self) -> None:
        for d in (
            self.logs_dir, self.workflow_logs_dir, self.outputs_dir,
            self.output_dir, self.state_dir, self.workflow_dir,
            self.data_store_dir, self.engineered_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# =========================================================================== #
# Structured logging
# =========================================================================== #
class WorkflowLogger:
    """
    Dual-channel logger:
      * human-readable  logs/workflow/workflow_<run_id>.log
      * structured JSONL logs/workflow/workflow_<run_id>.jsonl
      * failures-only   logs/workflow/failures_<run_id>.log
    Every record carries a *component* tag (yfinance, sheets, feature_engineering,
    metadata, model, forecast, rollover, retry, ...) so logs can be sliced per
    subsystem as required.  All three files plus stderr receive every event.
    """

    _LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

    def __init__(self, run_id: str, log_dir: Path) -> None:
        self.run_id = run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        self._human = open(log_dir / f"workflow_{run_id}.log", "a", encoding="utf-8")
        self._jsonl = open(log_dir / f"workflow_{run_id}.jsonl", "a", encoding="utf-8")
        self._fail = open(log_dir / f"failures_{run_id}.log", "a", encoding="utf-8")
        self.human_path = log_dir / f"workflow_{run_id}.log"
        self.jsonl_path = log_dir / f"workflow_{run_id}.jsonl"
        self.failures_path = log_dir / f"failures_{run_id}.log"
        self.counts = {k: 0 for k in self._LEVELS}

    def event(self, component: str, level: str, event: str, message: str, **fields: Any) -> None:
        level = level.upper()
        self.counts[level] = self.counts.get(level, 0) + 1
        ts = utc_iso()
        record = {
            "ts": ts, "run_id": self.run_id, "level": level,
            "component": component, "event": event, "message": message,
        }
        if fields:
            record["data"] = fields
        line_json = json.dumps(record, default=str)
        extra = ""
        if fields:
            extra = " | " + " ".join(f"{k}={_short(v)}" for k, v in fields.items())
        line_human = f"{ts} [{level:<8}] [{component:<20}] {message}{extra}"

        for stream in (self._human,):
            stream.write(line_human + "\n")
            stream.flush()
        self._jsonl.write(line_json + "\n")
        self._jsonl.flush()
        if level in ("ERROR", "CRITICAL", "WARNING"):
            self._fail.write(line_human + "\n")
            self._fail.flush()
        # mirror to stderr so Docker `compose logs` captures everything
        print(line_human, file=sys.stderr, flush=True)

    def banner(self, text: str) -> None:
        bar = "=" * 78
        for line in (bar, text, bar):
            self._human.write(line + "\n")
            print(line, file=sys.stderr, flush=True)
        self._human.flush()

    def close(self) -> None:
        for stream in (self._human, self._jsonl, self._fail):
            try:
                stream.close()
            except Exception:
                pass


def _short(value: Any, limit: int = 160) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# =========================================================================== #
# Stage result bookkeeping
# =========================================================================== #
@dataclass
class StageResult:
    index: int
    name: str
    status: str = "pending"          # pending | ok | skipped | degraded | failed
    gating: bool = False
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "name": self.name, "status": self.status,
            "gating": self.gating, "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "details": self.details, "error": self.error,
        }


# =========================================================================== #
# The orchestrator
# =========================================================================== #
class WorkflowOrchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_id: str = args.run_id or _run_id()
        self.live: bool = bool(args.live)
        self.dry_run: bool = not self.live
        self.cfg = WorkflowConfig.resolve(PROJECT_ROOT, args.google_credentials)
        self.cfg.ensure_dirs()
        self.log = WorkflowLogger(self.run_id, self.cfg.workflow_logs_dir)
        self.stages: List[StageResult] = []
        self.started_monotonic = time.monotonic()
        self.aborted = False
        # shared context passed between stages
        self.ctx: Dict[str, Any] = {
            "worksheets": self._parse_worksheets(args.worksheets),
            "yf_detection": {},
            "data_store": {},
            "engineered": {},
            "model_summary": {},
        }
        self._gspread_client = None  # cached, reused (API-quota friendly)

    # ----- small utilities ------------------------------------------------- #
    @staticmethod
    def _parse_worksheets(raw: Optional[str]) -> List[str]:
        if not raw:
            return []
        seen, out = set(), []
        for part in raw.split(","):
            name = part.strip()
            if name and name.upper() not in seen:
                seen.add(name.upper())
                out.append(name)
        return out

    def _subprocess_env(self) -> Dict[str, str]:
        """Environment for child processes — credentials wired automatically."""
        env = dict(os.environ)
        env["GOOGLE_APPLICATION_CREDENTIALS"] = str(self.cfg.credentials_path)
        env["GOOGLE_CREDENTIALS"] = str(self.cfg.credentials_path)
        env["SHEET_ID"] = self.cfg.test_sheet_id
        env["OPERATIONAL_SHEET_ID"] = self.cfg.test_sheet_id
        env["HISTORICAL_TRAINING_SHEET_ID"] = self.cfg.train_sheet_id
        env["BASE_DIR"] = str(self.cfg.base_dir)
        env["METADATA_PATH"] = str(self.cfg.metadata_path)
        env["MODEL_DIR"] = str(self.cfg.model_dir)
        env["OUTPUT_DIR"] = str(self.cfg.output_dir)
        env["PYTHONPATH"] = str(self.cfg.base_dir)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _gclient(self):
        """Return a cached, authorised gspread client (one per workflow run)."""
        if self._gspread_client is not None:
            return self._gspread_client
        from utils.google_auth import get_gspread_client
        self._gspread_client = retry_with_backoff(
            lambda: get_gspread_client(str(self.cfg.credentials_path)),
            label="gspread.authorize",
            attempts=self.cfg.google_retries,
            base_delay=self.cfg.retry_backoff,
            logger=self.log,
            component="sheets",
        )
        return self._gspread_client

    def _open_sheet(self, sheet_id: str):
        client = self._gclient()
        return retry_with_backoff(
            lambda: client.open_by_key(sheet_id),
            label=f"open_by_key({sheet_id[:12]}...)",
            attempts=self.cfg.google_retries,
            base_delay=self.cfg.retry_backoff,
            logger=self.log,
            component="sheets",
        )

    # ----- stage runner ---------------------------------------------------- #
    def _run_stage(self, index: int, name: str, fn: Callable[[StageResult], None]) -> StageResult:
        gating = index in GATING_STAGES
        result = StageResult(index=index, name=name, gating=gating,
                             started_at=utc_iso(), status="pending")
        self.stages.append(result)
        t0 = time.monotonic()
        self.log.banner(f"STAGE {index}/{STAGE_COUNT}  —  {name}"
                        f"   [{'GATE' if gating else 'soft'}]")
        try:
            fn(result)
            if result.status == "pending":
                result.status = "ok"
        except StageFailure as exc:
            result.status = "failed"
            result.error = str(exc)
            self.log.event("stage", "ERROR", "stage_failed",
                           f"Stage {index} ({name}) FAILED: {exc}")
        except Exception as exc:  # noqa: BLE001
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            self.log.event("stage", "CRITICAL", "stage_crashed",
                           f"Stage {index} ({name}) crashed: {exc}",
                           traceback=traceback.format_exc())
        finally:
            result.duration_seconds = time.monotonic() - t0
            result.finished_at = utc_iso()
            self.log.event("stage", "INFO", "stage_end",
                           f"Stage {index} ({name}) -> {result.status.upper()}",
                           duration_seconds=round(result.duration_seconds, 2))
        if result.status == "failed" and gating:
            self.aborted = True
        return result

    # ====================================================================== #
    # STAGE 1 — Startup validation
    # ====================================================================== #
    def stage_01_startup(self, r: StageResult) -> None:
        checks: Dict[str, Any] = {}
        critical: List[str] = []
        warnings: List[str] = []

        # 1. credentials file
        cred = self.cfg.credentials_path
        if cred.exists() and cred.stat().st_size > 0:
            try:
                cj = json.loads(cred.read_text(encoding="utf-8"))
                checks["credentials"] = {
                    "path": str(cred), "ok": True,
                    "service_account": cj.get("client_email", "<unknown>"),
                    "project_id": cj.get("project_id", "<unknown>"),
                }
            except Exception as exc:
                critical.append(f"credentials file is not valid JSON: {exc}")
        else:
            critical.append(f"credentials file missing: {cred}")

        # 2. required directories exist & writable
        dir_status = {}
        for label, d in (
            ("logs", self.cfg.logs_dir), ("outputs", self.cfg.outputs_dir),
            ("main_inference", self.cfg.output_dir), ("state", self.cfg.state_dir),
            ("workflow", self.cfg.workflow_dir),
        ):
            d.mkdir(parents=True, exist_ok=True)
            writable = os.access(d, os.W_OK)
            dir_status[label] = {"path": str(d), "writable": writable}
            if not writable:
                critical.append(f"directory not writable: {d}")
        checks["directories"] = dir_status

        # 3. model files exist (PROTECTED dir — read-only check)
        model_status = {}
        for fname in ("Dense.pt", "LSTM.pt", "Transformer.pt"):
            fp = self.cfg.model_dir / fname
            exists = fp.exists()
            size = fp.stat().st_size if exists else 0
            model_status[fname] = {"exists": exists, "size_bytes": size}
            if not exists:
                critical.append(f"missing model checkpoint: {fp}")
            elif size < 1024:
                warnings.append(f"model checkpoint suspiciously small: {fp} ({size}b)")
        checks["model_files"] = model_status

        # 4. metadata exists & schema-valid (self-healing)
        try:
            from app.pipeline.metadata import (
                initialize_metadata_if_missing, repair_metadata_if_corrupted,
                safe_load_metadata, validate_metadata_schema,
            )
            created = initialize_metadata_if_missing(self.cfg.metadata_path)
            repaired = repair_metadata_if_corrupted(self.cfg.metadata_path)
            meta = safe_load_metadata(self.cfg.metadata_path)
            errors = validate_metadata_schema(meta)
            checks["metadata"] = {
                "path": str(self.cfg.metadata_path), "created": created,
                "repaired": repaired, "schema_errors": errors,
                "feature_count": meta.get("feature_count"),
                "seq_len": meta.get("seq_len"), "horizons": meta.get("horizons"),
            }
            if errors:
                critical.append(f"metadata schema invalid after repair: {errors}")
        except Exception as exc:
            critical.append(f"metadata validation failed: {exc}")

        # 5. workbook / training corpus presence (informational)
        wb_train = self.cfg.base_dir / "Data" / "nse_stock_data_train.xlsx"
        checks["workbook"] = {
            "training_corpus": str(wb_train), "exists": wb_train.exists(),
        }

        # 6. Docker volume mount paths (informational — present when in Docker)
        checks["docker_volumes"] = {
            p: Path(p).exists() for p in (
                "/app/logs", "/app/state", "/app/outputs/main_inference",
                "/app/credentials/Credentials_New.json",
            )
        }
        checks["running_in_docker"] = Path("/.dockerenv").exists()

        # 7. real write-probe on every directory the workflow actually writes
        #    to (catches Docker read-only mounts). The probe creates and then
        #    removes a uniquely-named file — never a dotfile, never the bare
        #    outputs/ root, so it is robust on bind-mounted filesystems.
        # Writability is proven by a successful *write*. A blocked cleanup
        # (unlink) is tolerated — the workflow only ever creates files via
        # atomic create+rename and its own cleanup is best-effort.
        write_tests: Dict[str, str] = {}
        probe_token = f"workflow_write_probe_{os.getpid()}_{int(time.time() * 1000)}"
        for label, d, severity in (
            ("outputs/workflow", self.cfg.workflow_dir, "CRITICAL"),
            ("outputs/main_inference", self.cfg.output_dir, "CRITICAL"),
            ("logs", self.cfg.logs_dir, "CRITICAL"),
            ("state", self.cfg.state_dir, "CRITICAL"),
            ("outputs (root)", self.cfg.outputs_dir, "WARNING"),
        ):
            probe = d / probe_token
            try:
                probe.write_text(utc_iso(), encoding="utf-8")
            except OSError as exc:
                write_tests[label] = f"FAILED: {exc}"
                msg = f"directory not writable ({label}): {exc}"
                (critical if severity == "CRITICAL" else warnings).append(msg)
                continue
            try:
                probe.unlink()
                write_tests[label] = "passed"
            except OSError as exc:
                write_tests[label] = f"passed (write ok; cleanup blocked: {exc})"
        checks["write_tests"] = write_tests

        # 8. repository startup self-check (best-effort, never fatal here)
        try:
            from app.config.settings import get_settings
            from app.pipeline.startup import run_startup_checks
            run_startup_checks(get_settings(), strict=False)
            checks["repo_startup_checks"] = "passed"
        except Exception as exc:
            warnings.append(f"repo run_startup_checks reported: {exc}")
            checks["repo_startup_checks"] = f"warning: {exc}"

        # 9. Google API connectivity — CRITICAL in --live, WARNING in dry-run
        conn_ok, conn_msg = self._probe_google()
        checks["google_connectivity"] = {"ok": conn_ok, "detail": conn_msg}
        if not conn_ok:
            if self.live:
                critical.append(f"Google API connectivity failed: {conn_msg}")
            else:
                warnings.append(f"Google API not reachable (dry-run tolerated): {conn_msg}")

        r.details = {"checks": checks, "critical": critical, "warnings": warnings,
                     "mode": "live" if self.live else "dry-run"}
        for w in warnings:
            self.log.event("startup", "WARNING", "startup_warning", w)
        for c in critical:
            self.log.event("startup", "CRITICAL", "startup_critical", c)
        if critical:
            raise StageFailure(f"{len(critical)} critical startup issue(s): {critical}")
        self.log.event("startup", "INFO", "startup_ok",
                       "All critical startup checks passed",
                       warnings=len(warnings))

    def _probe_google(self) -> Tuple[bool, str]:
        """Lightweight connectivity probe — DNS + credential load + open sheet."""
        try:
            socket.getaddrinfo("sheets.googleapis.com", 443)
        except OSError as exc:
            return False, f"DNS/network unreachable: {exc}"
        try:
            sh = self._open_sheet(self.cfg.test_sheet_id)
            title = getattr(sh, "title", "<sheet>")
            return True, f"opened operational sheet '{title}'"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    # ====================================================================== #
    # STAGE 2 — Check YFinance for new data
    # ====================================================================== #
    def stage_02_check_yfinance(self, r: StageResult) -> None:
        sh = self._open_sheet(self.cfg.test_sheet_id)
        all_ws = retry_with_backoff(
            sh.worksheets, label="list worksheets",
            attempts=self.cfg.google_retries, base_delay=self.cfg.retry_backoff,
            logger=self.log, component="sheets",
        )
        discovered = [w.title for w in all_ws
                      if w.title.strip().upper() not in NON_STOCK_TABS]
        targets = self.ctx["worksheets"] or discovered
        # keep only worksheets that actually exist
        existing = {w.title.upper(): w for w in all_ws}
        targets = [t for t in targets if t.upper() in existing]
        if not targets:
            raise StageFailure("no target stock worksheets found in operational sheet")
        self.ctx["worksheets"] = targets
        self.ctx["_worksheet_objs"] = {t: existing[t.upper()] for t in targets}

        max_data_date = self._safe_market_date()
        detection: Dict[str, Any] = {}
        new_data_total = 0
        for title in targets:
            ws = existing[title.upper()]
            try:
                last_date = self._worksheet_last_date(ws)
            except Exception as exc:  # noqa: BLE001
                detection[title] = {"status": "read_error", "error": str(exc)}
                self.log.event("yfinance", "WARNING", "sheet_read_error",
                               f"{title}: could not read last date: {exc}")
                continue
            if last_date is None:
                detection[title] = {"status": "empty_or_no_dates",
                                    "new_data_expected": True}
                new_data_total += 1
                continue
            gap_days = (max_data_date - last_date).days
            probe = self._yfinance_probe(title, last_date, max_data_date)
            has_new = probe.get("rows", 0) > 0 if probe.get("ok") else gap_days > 0
            detection[title] = {
                "status": "checked",
                "last_date": last_date.isoformat(),
                "market_date": max_data_date.isoformat(),
                "gap_days": gap_days,
                "yfinance_probe": probe,
                "new_data_expected": bool(has_new),
            }
            if has_new:
                new_data_total += 1
            self.log.event("yfinance", "INFO", "detection",
                           f"{title}: last={last_date} gap={gap_days}d "
                           f"new_data={has_new}")

        self.ctx["yf_detection"] = detection
        r.details = {
            "discovered_worksheets": len(discovered),
            "target_worksheets": targets,
            "worksheets_with_new_data": new_data_total,
            "market_data_date": max_data_date.isoformat(),
            "detection": detection,
        }
        self.log.event("yfinance", "INFO", "summary",
                       f"{new_data_total}/{len(targets)} worksheet(s) expect new data")

    @staticmethod
    def _safe_market_date() -> date:
        """Last safely-complete trading date (IST, 16:00 cutoff)."""
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            cutoff = now.replace(hour=16, minute=0, second=0, microsecond=0)
            d = now.date() if now >= cutoff else now.date() - timedelta(days=1)
        except Exception:
            d = utc_now().date() - timedelta(days=1)
        # step back over weekends
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d

    def _worksheet_last_date(self, ws) -> Optional[date]:
        values = retry_with_backoff(
            ws.get_all_values, label=f"{ws.title}: get_all_values",
            attempts=self.cfg.google_retries, base_delay=self.cfg.retry_backoff,
            logger=self.log, component="sheets",
        )
        if not values or len(values) < 2:
            return None
        headers = [str(h).strip().lower() for h in values[0]]
        date_idx = next(
            (i for i, h in enumerate(headers)
             if h in ("date", "date_str", "datetime")), None,
        )
        if date_idx is None:
            return None
        latest: Optional[date] = None
        for row in values[1:]:
            if date_idx >= len(row):
                continue
            parsed = self._parse_date(row[date_idx])
            if parsed and (latest is None or parsed > latest):
                latest = parsed
        return latest

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        text = str(value).strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
                    "%Y-%m-%d %H:%M:%S", "%m/%d/%Y"):
            try:
                return datetime.strptime(text[:19], fmt).date()
            except ValueError:
                continue
        try:
            import pandas as pd
            ts = pd.to_datetime(text, errors="coerce")
            return None if ts is None or pd.isna(ts) else ts.date()
        except Exception:
            return None

    def _yfinance_probe(self, title: str, last_date: date, market_date: date) -> Dict[str, Any]:
        """Read-only probe: does yfinance have rows after *last_date*?"""
        if last_date >= market_date:
            return {"ok": True, "rows": 0, "note": "sheet already current"}
        try:
            import yfinance as yf
            symbol = f"{title.strip().upper()}.NS"
            start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            end = (market_date + timedelta(days=1)).strftime("%Y-%m-%d")
            df = retry_with_backoff(
                lambda: yf.download(symbol, start=start, end=end, interval="1d",
                                    progress=False, auto_adjust=False),
                label=f"{symbol}: yfinance probe",
                attempts=self.cfg.google_retries, base_delay=self.cfg.retry_backoff,
                logger=self.log, component="yfinance",
            )
            rows = 0 if df is None else int(len(df))
            return {"ok": True, "rows": rows, "symbol": symbol}
        except ImportError:
            return {"ok": False, "error": "yfinance not installed",
                    "fallback": "date-gap heuristic"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "fallback": "date-gap heuristic"}

    # ====================================================================== #
    # STAGE 3 — Download / append new market data  (Data_update.py)
    # ====================================================================== #
    def stage_03_download_append(self, r: StageResult) -> None:
        if self.dry_run:
            expected = [k for k, v in self.ctx["yf_detection"].items()
                        if v.get("new_data_expected")]
            r.status = "skipped"
            r.details = {
                "skipped_reason": "dry-run — Data_update.py performs live sheet "
                                  "appends and is not executed without --live",
                "worksheets_that_would_be_updated": expected,
            }
            self.log.event("yfinance", "INFO", "dry_run_skip",
                           f"dry-run: would append new data to {len(expected)} worksheet(s)")
            return

        cmd = [
            sys.executable, "Data_update.py",
            "--sheet-id", self.cfg.test_sheet_id,
            "--google-credentials", str(self.cfg.credentials_path),
            "--start-date", self.cfg.update_start_date,
            "--interval", self.cfg.update_interval,
        ]
        if self.ctx["worksheets"]:
            cmd += ["--worksheets", ",".join(self.ctx["worksheets"])]

        proc = self._run_subprocess(cmd, "Data_update.py", component="yfinance",
                                    retries=self.cfg.subprocess_retries)
        parsed = _parse_last_json(proc["stdout"]) or {}
        r.details = {
            "command": cmd,
            "returncode": proc["returncode"],
            "duration_seconds": proc["duration"],
            "update_result": parsed,
            "rows_added": parsed.get("rows_added"),
            "stocks_updated": parsed.get("stocks_updated"),
        }
        self.ctx["data_update_result"] = parsed
        status = parsed.get("status")
        if proc["returncode"] != 0 or status == "error":
            raise StageFailure(
                f"Data_update.py failed (returncode={proc['returncode']}, "
                f"status={status}): {parsed.get('error') or proc['stderr'][-400:]}"
            )
        if status == "partial_error":
            r.status = "degraded"
            self.log.event("yfinance", "WARNING", "partial_update",
                           f"Data_update.py partial_error — "
                           f"{parsed.get('failures')} worksheet failure(s)")
        self.log.event("yfinance", "INFO", "append_done",
                       f"Data_update.py status={status} "
                       f"rows_added={parsed.get('rows_added')}")

    # ====================================================================== #
    # STAGE 4 — Update local workbook / data store
    # ====================================================================== #
    def stage_04_local_store(self, r: StageResult) -> None:
        """Snapshot every target worksheet to a local CSV store.

        The pipeline is Google-Sheets-sourced, so the 'local workbook' is a
        deterministic on-disk mirror used for integrity checks (stage 6) and
        feature engineering (stage 7). Writes are atomic.
        """
        import pandas as pd  # local import keeps orchestrator import-light
        ws_objs = self.ctx.get("_worksheet_objs", {})
        store: Dict[str, Any] = {}
        snap_dir = self.cfg.data_store_dir
        total_rows = 0
        for title in self.ctx["worksheets"]:
            ws = ws_objs.get(title)
            if ws is None:
                store[title] = {"status": "missing_worksheet"}
                continue
            values = retry_with_backoff(
                ws.get_all_values, label=f"{title}: snapshot",
                attempts=self.cfg.google_retries, base_delay=self.cfg.retry_backoff,
                logger=self.log, component="sheets",
            )
            if not values:
                store[title] = {"status": "empty"}
                continue
            headers = [str(h).strip() for h in values[0]]
            width = len(headers)
            rows = [list(row) + [""] * (width - len(row)) for row in values[1:]]
            df = pd.DataFrame([row[:width] for row in rows], columns=headers)
            csv_path = snap_dir / f"{title}.csv"
            tmp = csv_path.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False)
            os.replace(tmp, csv_path)
            total_rows += len(df)
            store[title] = {
                "status": "snapshotted", "rows": len(df),
                "columns": len(headers), "path": str(csv_path),
            }
        self.ctx["data_store"] = store
        r.details = {"store_dir": str(snap_dir),
                     "worksheets_snapshotted": sum(
                         1 for v in store.values() if v.get("status") == "snapshotted"),
                     "total_rows": total_rows, "per_worksheet": store}
        self.log.event("sheets", "INFO", "local_store",
                       f"snapshotted {r.details['worksheets_snapshotted']} worksheet(s), "
                       f"{total_rows} rows -> {snap_dir}")

    # ====================================================================== #
    # STAGE 5 — Sync / append new rows into Google Sheets
    # ====================================================================== #
    def stage_05_sheet_sync(self, r: StageResult) -> None:
        """Reconcile the operational sheet against what stage 3 reported.

        In live mode Data_update.py already performed the append; this stage
        verifies it landed (no duplicates, schema preserved). In dry-run it is
        a simulation report.
        """
        if self.dry_run:
            r.status = "skipped"
            r.details = {"skipped_reason": "dry-run — no rows written; "
                                           "Google Sheet append is simulated"}
            self.log.event("sheets", "INFO", "dry_run_skip",
                           "dry-run: Google Sheets sync simulated")
            return

        upd = self.ctx.get("data_update_result", {})
        per_sheet = upd.get("worksheets", []) if isinstance(upd, dict) else []
        verified: Dict[str, Any] = {}
        ws_objs = self.ctx.get("_worksheet_objs", {})
        for title in self.ctx["worksheets"]:
            ws = ws_objs.get(title)
            if ws is None:
                continue
            dup = self._duplicate_dates(ws)
            verified[title] = {"duplicate_dates": dup}
            if dup:
                self.log.event("sheets", "WARNING", "duplicate_dates",
                               f"{title}: {len(dup)} duplicate date(s) detected: "
                               f"{dup[:5]}")
        r.details = {"data_update_per_sheet": per_sheet, "verification": verified,
                     "rows_added": upd.get("rows_added")}
        any_dups = any(v["duplicate_dates"] for v in verified.values())
        if any_dups:
            r.status = "degraded"
        self.log.event("sheets", "INFO", "sync_verified",
                       f"sheet sync verified for {len(verified)} worksheet(s)")

    # ====================================================================== #
    # STAGE 6 — Validate workbook / sheet integrity
    # ====================================================================== #
    def stage_06_validate_integrity(self, r: StageResult) -> None:
        import pandas as pd
        store = self.ctx["data_store"]
        report: Dict[str, Any] = {}
        problems: List[str] = []
        healthy = 0
        for title in self.ctx["worksheets"]:
            entry = store.get(title, {})
            csv_path = entry.get("path")
            if not csv_path or not Path(csv_path).exists():
                report[title] = {"ok": False, "reason": "snapshot missing"}
                problems.append(f"{title}: snapshot missing")
                continue
            df = pd.read_csv(csv_path)
            cols_lower = {str(c).strip().lower(): c for c in df.columns}
            missing = [c for c in REQUIRED_SHEET_COLUMNS
                       if c.lower() not in cols_lower]
            date_col = next((cols_lower[k] for k in ("date", "date_str", "datetime")
                             if k in cols_lower), None)
            issues: List[str] = []
            if missing:
                issues.append(f"missing columns {missing}")
            n_dates = n_dup = 0
            monotonic = True
            if date_col is not None:
                parsed = pd.to_datetime(df[date_col], errors="coerce")
                valid = parsed.dropna()
                n_dates = int(len(valid))
                n_dup = int(valid.duplicated().sum())
                if n_dup:
                    issues.append(f"{n_dup} duplicate date(s)")
                monotonic = bool(valid.is_monotonic_increasing)
                if not monotonic:
                    issues.append("dates not chronologically ordered")
            else:
                issues.append("no date column")
            ok = not issues
            report[title] = {
                "ok": ok, "rows": int(len(df)), "valid_dates": n_dates,
                "duplicate_dates": n_dup, "chronological": monotonic,
                "issues": issues,
            }
            if ok:
                healthy += 1
            else:
                problems.extend(f"{title}: {i}" for i in issues)
        r.details = {"worksheets_checked": len(self.ctx["worksheets"]),
                     "healthy": healthy, "problems": problems,
                     "per_worksheet": report}
        for p in problems:
            self.log.event("sheets", "ERROR", "integrity_problem", p)
        if healthy == 0:
            raise StageFailure(f"no worksheet passed integrity validation: {problems}")
        if problems:
            r.status = "degraded"
            self.log.event("sheets", "WARNING", "integrity_degraded",
                           f"{healthy}/{len(self.ctx['worksheets'])} worksheet(s) healthy")
        else:
            self.log.event("sheets", "INFO", "integrity_ok",
                           f"all {healthy} worksheet(s) passed integrity validation")

    # ====================================================================== #
    # STAGE 7 — Run Feature Engineering
    # ====================================================================== #
    def stage_07_feature_engineering(self, r: StageResult) -> None:
        """Explicit FE gate.

        Runs Feature_Engineering.compute_indicators on every worksheet snapshot
        and persists the engineered frames to outputs/workflow/engineered/.
        This GUARANTEES feature engineering succeeds before any model inference
        begins. main.py later re-runs FE per-symbol for inference — this stage
        is the upfront gate that proves FE works on the current data.
        """
        import pandas as pd
        try:
            from Feature_Engineering import (
                compute_indicators, detect_date_column, ensure_date_column,
            )
        except Exception as exc:  # noqa: BLE001
            raise StageFailure(f"cannot import Feature_Engineering: {exc}") from exc

        store = self.ctx["data_store"]
        engineered: Dict[str, Any] = {}
        ok_count = 0
        for title in self.ctx["worksheets"]:
            entry = store.get(title, {})
            csv_path = entry.get("path")
            if not csv_path or not Path(csv_path).exists():
                engineered[title] = {"status": "skipped", "reason": "no snapshot"}
                continue
            try:
                df = pd.read_csv(csv_path)
                # numeric coercion for OHLCV-style columns
                for col in df.columns:
                    if str(col).strip().lower() in (
                        "open", "high", "low", "close", "adj close", "volume",
                    ):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                date_col = detect_date_column(df)
                if date_col is not None:
                    df = ensure_date_column(df, date_col)
                feat = compute_indicators(df)
                out_path = self.cfg.engineered_dir / f"{title}.parquet"
                tmp = out_path.with_suffix(".parquet.tmp")
                try:
                    feat.to_parquet(tmp, index=False)
                except Exception:
                    # parquet engine unavailable — fall back to CSV
                    out_path = self.cfg.engineered_dir / f"{title}.csv"
                    tmp = out_path.with_suffix(".csv.tmp")
                    feat.to_csv(tmp, index=False)
                os.replace(tmp, out_path)
                engineered[title] = {
                    "status": "ok", "rows": int(len(feat)),
                    "columns": int(len(feat.columns)), "path": str(out_path),
                }
                ok_count += 1
                self.log.event("feature_engineering", "INFO", "fe_ok",
                               f"{title}: {len(feat)} rows x {len(feat.columns)} cols")
            except Exception as exc:  # noqa: BLE001
                engineered[title] = {"status": "failed",
                                     "error": f"{type(exc).__name__}: {exc}"}
                self.log.event("feature_engineering", "ERROR", "fe_failed",
                               f"{title}: feature engineering failed: {exc}")
        self.ctx["engineered"] = engineered
        r.details = {"worksheets": len(self.ctx["worksheets"]),
                     "succeeded": ok_count, "per_worksheet": engineered,
                     "engineered_dir": str(self.cfg.engineered_dir)}
        if ok_count == 0:
            raise StageFailure(
                "feature engineering failed for ALL worksheets — "
                "model inference will NOT proceed")
        if ok_count < len(self.ctx["worksheets"]):
            r.status = "degraded"
        self.log.event("feature_engineering", "INFO", "fe_summary",
                       f"feature engineering succeeded for {ok_count}/"
                       f"{len(self.ctx['worksheets'])} worksheet(s)")

    # ====================================================================== #
    # STAGE 8 — Validate engineered features
    # ====================================================================== #
    def stage_08_validate_features(self, r: StageResult) -> None:
        import pandas as pd
        from app.pipeline.metadata import safe_load_metadata
        meta = safe_load_metadata(self.cfg.metadata_path)
        required_features = list(meta.get("feature_columns", []))
        label_cols = list(meta.get("label_columns", []))
        engineered = self.ctx["engineered"]
        report: Dict[str, Any] = {}
        valid = 0
        problems: List[str] = []
        for title, info in engineered.items():
            if info.get("status") != "ok":
                continue
            path = Path(info["path"])
            try:
                df = pd.read_parquet(path) if path.suffix == ".parquet" \
                    else pd.read_csv(path)
            except Exception as exc:  # noqa: BLE001
                report[title] = {"ok": False, "reason": f"unreadable: {exc}"}
                problems.append(f"{title}: engineered file unreadable")
                continue
            cols = set(map(str, df.columns))
            missing_feats = [c for c in required_features if c not in cols]
            has_labels = any(c in cols for c in label_cols) if label_cols else None
            # column is "alive" if its last 5 rows contain at least one finite value
            stale_cols = []
            for c in required_features:
                if c in cols:
                    tail = pd.to_numeric(df[c].tail(5), errors="coerce")
                    if tail.notna().sum() == 0:
                        stale_cols.append(c)
            issues = []
            if missing_feats:
                issues.append(f"missing feature columns: {missing_feats}")
            if stale_cols:
                issues.append(f"all-NaN tail in: {stale_cols}")
            if len(df) == 0:
                issues.append("engineered frame is empty")
            ok = not issues
            report[title] = {
                "ok": ok, "rows": int(len(df)),
                "feature_columns_present": len(required_features) - len(missing_feats),
                "feature_columns_required": len(required_features),
                "label_columns_present": has_labels,
                "issues": issues,
            }
            if ok:
                valid += 1
            else:
                problems.extend(f"{title}: {i}" for i in issues)
        r.details = {"required_feature_count": len(required_features),
                     "valid_worksheets": valid, "problems": problems,
                     "per_worksheet": report}
        for p in problems:
            self.log.event("feature_engineering", "ERROR", "feature_validation",
                            p)
        if valid == 0:
            raise StageFailure(
                "no worksheet produced valid engineered features matching "
                f"metadata schema ({len(required_features)} features) — "
                "aborting before model inference")
        if problems:
            r.status = "degraded"
        self.log.event("feature_engineering", "INFO", "feature_validation_ok",
                       f"engineered features validated for {valid} worksheet(s)")

    # ====================================================================== #
    # STAGE 9 — Update metadata
    # ====================================================================== #
    def stage_09_update_metadata(self, r: StageResult) -> None:
        from app.pipeline.metadata import (
            repair_metadata_if_corrupted, safe_load_metadata,
            safe_save_metadata, validate_metadata_schema,
        )
        repair_metadata_if_corrupted(self.cfg.metadata_path)
        meta = safe_load_metadata(self.cfg.metadata_path)
        errors = validate_metadata_schema(meta)
        if errors:
            raise StageFailure(f"metadata schema invalid before update: {errors}")
        # record FE provenance under a namespaced key (never touches the
        # model-architecture keys the inference pipeline hard-depends on).
        wf = meta.setdefault("workflow", {})
        wf["last_run_id"] = self.run_id
        wf["last_feature_engineering_at"] = utc_iso()
        wf["feature_engineering_worksheets"] = [
            t for t, v in self.ctx["engineered"].items()
            if v.get("status") == "ok"
        ]
        wf["mode"] = "live" if self.live else "dry-run"
        safe_save_metadata(self.cfg.metadata_path, meta)
        # confirm the write is still schema-valid
        reloaded = safe_load_metadata(self.cfg.metadata_path)
        post_errors = validate_metadata_schema(reloaded)
        if post_errors:
            raise StageFailure(f"metadata corrupted by update: {post_errors}")
        r.details = {"metadata_path": str(self.cfg.metadata_path),
                     "schema_valid": True,
                     "feature_count": reloaded.get("feature_count"),
                     "workflow_provenance": wf}
        self.log.event("metadata", "INFO", "metadata_updated",
                       "metadata updated atomically + backup written; "
                       "schema re-validated")

    # ====================================================================== #
    # STAGE 10 — Run model pipeline (main.py)
    # ====================================================================== #
    def stage_10_model_pipeline(self, r: StageResult) -> None:
        cmd = [
            sys.executable, "main.py",
            "--source", "google",
            "--sheet-id", self.cfg.test_sheet_id,
            "--google-credentials", str(self.cfg.credentials_path),
            "--model-dir", str(self.cfg.model_dir),
            "--metadata", str(self.cfg.metadata_path),
            "--output-dir", str(self.cfg.output_dir),
            "--device", self.cfg.device,
        ]
        if self.ctx["worksheets"]:
            cmd += ["--worksheets", ",".join(self.ctx["worksheets"])]
        if self.dry_run:
            cmd.append("--dry-run")

        proc = self._run_subprocess(cmd, "main.py", component="model",
                                    retries=self.cfg.subprocess_retries)
        parsed = _parse_last_json(proc["stdout"]) or {}
        self.ctx["model_summary"] = parsed
        r.details = {
            "command": cmd, "returncode": proc["returncode"],
            "duration_seconds": proc["duration"],
            "pipeline_status": parsed.get("status"),
            "processed_worksheet_count": parsed.get("processed_worksheet_count"),
            "predicted_row_count": parsed.get("predicted_row_count"),
            "metrics_path": parsed.get("metrics_path"),
            "predictions_csv_path": parsed.get("predictions_csv_path"),
            "skipped": parsed.get("skipped", []),
        }
        if proc["returncode"] != 0:
            raise StageFailure(
                f"main.py model pipeline failed (returncode={proc['returncode']}): "
                f"{proc['stderr'][-500:]}")
        if not parsed:
            raise StageFailure(
                "main.py produced no parseable JSON summary — "
                "inference completion cannot be confirmed")
        if parsed.get("status") == "error":
            raise StageFailure(
                f"main.py reported status=error: {parsed.get('error', 'unknown')}")
        self.log.event("model", "INFO", "model_pipeline_done",
                       f"main.py status={parsed.get('status')} "
                       f"predicted_rows={parsed.get('predicted_row_count')}")

    # ====================================================================== #
    # STAGE 11 — Generate forecasts / predictions
    # ====================================================================== #
    def stage_11_generate_forecasts(self, r: StageResult) -> None:
        summary = self.ctx["model_summary"]
        preds = summary.get("predictions_csv_path")
        metrics = summary.get("metrics_path")
        artifacts: Dict[str, Any] = {}
        problems: List[str] = []
        for label, path in (("predictions_csv", preds), ("metrics_json", metrics)):
            if path and Path(path).exists():
                st = Path(path).stat()
                age = time.time() - st.st_mtime
                artifacts[label] = {"path": path, "size_bytes": st.st_size,
                                    "age_seconds": round(age, 1)}
                if age > 3600:
                    problems.append(f"{label} looks stale ({age/60:.0f} min old)")
            else:
                artifacts[label] = {"path": path, "exists": False}
                problems.append(f"{label} not found")
        predicted = summary.get("predicted_row_count", 0) or 0
        r.details = {"artifacts": artifacts, "predicted_row_count": predicted,
                     "problems": problems}
        if problems:
            r.status = "degraded"
            for p in problems:
                self.log.event("forecast", "WARNING", "forecast_artifact", p)
        self.log.event("forecast", "INFO", "forecasts_generated",
                       f"{predicted} forecast row(s) generated")

    # ====================================================================== #
    # STAGE 12 — Push forecast results back into Google Sheets
    # ====================================================================== #
    def stage_12_push_forecasts(self, r: StageResult) -> None:
        summary = self.ctx["model_summary"]
        if self.dry_run:
            r.status = "skipped"
            r.details = {"skipped_reason": "dry-run — main.py --dry-run computed "
                                           "forecasts without writing to sheets",
                         "would_write": bool(summary.get("predicted_row_count"))}
            self.log.event("forecast", "INFO", "dry_run_skip",
                           "dry-run: forecast push to Google Sheets simulated")
            return
        written = summary.get("sheet_updates_written")
        errors = summary.get("sheet_update_errors", []) or []
        r.details = {"sheet_updates_written": written,
                     "sheet_update_errors": errors}
        if errors:
            r.status = "degraded"
            for e in errors:
                self.log.event("forecast", "ERROR", "forecast_push_error",
                               f"forecast push error: {e}")
        # independent duplicate-forecast check
        dup_report = {}
        ws_objs = self.ctx.get("_worksheet_objs", {})
        for title, ws in ws_objs.items():
            try:
                dups = self._duplicate_dates(ws)
                if dups:
                    dup_report[title] = dups[:10]
            except Exception:
                pass
        if dup_report:
            r.status = "degraded"
            r.details["duplicate_date_rows"] = dup_report
            self.log.event("forecast", "WARNING", "duplicate_forecast_rows",
                           f"duplicate dates detected post-push: {list(dup_report)}")
        self.log.event("forecast", "INFO", "forecasts_pushed",
                       f"forecast push complete (updates_written={written})")

    # ====================================================================== #
    # STAGE 13 — Train/test rollover validation
    # ====================================================================== #
    def stage_13_rollover(self, r: StageResult) -> None:
        """The rollover itself runs INSIDE main.py (cleanup_google_sheet_latest_rows
        -> sheet_archival.archive_old_rows_for_worksheet). This stage independently
        VALIDATES it: TEST <= 30 rows, TRAIN grew, chronology + dedup preserved.
        """
        summary = self.ctx["model_summary"]
        cleanup = summary.get("sheet_cleanup_results", []) or []
        if self.dry_run:
            r.status = "skipped"
            r.details = {"skipped_reason": "dry-run — rollover archives rows and is "
                                           "skipped by main.py --dry-run",
                         "rolling_window": ROLLING_TEST_ROWS}
            self.log.event("rollover", "INFO", "dry_run_skip",
                           "dry-run: train/test rollover simulated")
            return

        validation: Dict[str, Any] = {}
        problems: List[str] = []
        # TEST sheet — each worksheet must be within the rolling window
        for title, ws in self.ctx.get("_worksheet_objs", {}).items():
            try:
                values = retry_with_backoff(
                    ws.get_all_values, label=f"{title}: rollover check",
                    attempts=self.cfg.google_retries,
                    base_delay=self.cfg.retry_backoff,
                    logger=self.log, component="rollover",
                )
                data_rows = max(0, len(values) - 1)
                within = data_rows <= ROLLING_TEST_ROWS + 2  # small slack
                validation[title] = {"test_rows": data_rows,
                                     "within_window": within}
                if not within:
                    problems.append(
                        f"{title}: TEST has {data_rows} rows "
                        f"(> {ROLLING_TEST_ROWS} rolling window)")
            except Exception as exc:  # noqa: BLE001
                validation[title] = {"error": str(exc)}
        # TRAIN sheet — chronology + duplicate check on archived worksheets
        train_report: Dict[str, Any] = {}
        try:
            train_sh = self._open_sheet(self.cfg.train_sheet_id)
            train_ws = {w.title.upper(): w for w in train_sh.worksheets()}
            for title in self.ctx.get("_worksheet_objs", {}):
                tws = train_ws.get(title.upper())
                if tws is None:
                    train_report[title] = {"present": False}
                    continue
                dups = self._duplicate_dates(tws)
                chrono = self._is_chronological(tws)
                train_report[title] = {"present": True,
                                       "duplicate_dates": len(dups),
                                       "chronological": chrono}
                if dups:
                    problems.append(f"TRAIN/{title}: {len(dups)} duplicate date(s)")
                if not chrono:
                    problems.append(f"TRAIN/{title}: not chronological")
        except Exception as exc:  # noqa: BLE001
            train_report["_error"] = str(exc)
            self.log.event("rollover", "WARNING", "train_sheet_check",
                           f"could not validate TRAIN sheet: {exc}")
        r.details = {"main_py_cleanup_results": cleanup,
                     "test_sheet_validation": validation,
                     "train_sheet_validation": train_report,
                     "rolling_window": ROLLING_TEST_ROWS,
                     "problems": problems}
        for p in problems:
            self.log.event("rollover", "ERROR", "rollover_problem", p)
        if problems:
            r.status = "degraded"
        self.log.event("rollover", "INFO", "rollover_validated",
                       f"train/test rollover validated "
                       f"({len(problems)} problem(s))")

    # ====================================================================== #
    # STAGE 14 — Validate final outputs
    # ====================================================================== #
    def stage_14_validate_final(self, r: StageResult) -> None:
        summary = self.ctx["model_summary"]
        checks: Dict[str, Any] = {}
        problems: List[str] = []

        # metadata still valid + freshly updated
        try:
            from app.pipeline.metadata import (safe_load_metadata,
                                                validate_metadata_schema)
            meta = safe_load_metadata(self.cfg.metadata_path)
            errs = validate_metadata_schema(meta)
            checks["metadata_schema_valid"] = not errs
            checks["metadata_last_run_id"] = meta.get("workflow", {}).get("last_run_id")
            if errs:
                problems.append(f"metadata schema errors: {errs}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"metadata re-check failed: {exc}")

        # inference artifacts exist
        for label in ("predictions_csv_path", "metrics_path"):
            path = summary.get(label)
            exists = bool(path and Path(path).exists())
            checks[label] = {"path": path, "exists": exists}
            if not exists:
                problems.append(f"missing inference artifact: {label}")

        # pipeline reported a non-error status
        checks["pipeline_status"] = summary.get("status")
        if summary.get("status") == "error":
            problems.append("model pipeline status=error")

        # incomplete-workflow detection: every non-skipped stage finished
        unfinished = [s.name for s in self.stages
                      if s.status not in ("ok", "skipped", "degraded")]
        checks["unfinished_stages"] = unfinished
        if unfinished:
            problems.append(f"unfinished stages: {unfinished}")

        r.details = {"checks": checks, "problems": problems}
        for p in problems:
            self.log.event("validation", "ERROR", "final_validation", p)
        if problems:
            r.status = "degraded"
            self.log.event("validation", "WARNING", "final_validation_degraded",
                           f"{len(problems)} final-validation problem(s)")
        else:
            self.log.event("validation", "INFO", "final_validation_ok",
                           "all final-output validations passed")

    # ====================================================================== #
    # STAGE 15 — Log workflow summary
    # ====================================================================== #
    def stage_15_summary(self, r: StageResult) -> None:
        summary = self.build_summary()
        out_path = self.cfg.workflow_dir / f"run_summary_{self.run_id}.json"
        atomic_write_json(out_path, summary)
        latest = self.cfg.workflow_dir / "run_summary_latest.json"
        atomic_write_json(latest, summary)
        r.details = {"summary_path": str(out_path),
                     "latest_path": str(latest),
                     "overall_status": summary["overall_status"]}
        self.log.event("summary", "INFO", "summary_written",
                       f"workflow summary written -> {out_path}")
        # human-readable timing table
        self.log.banner("WORKFLOW TIMING SUMMARY")
        for s in self.stages:
            self.log.event("summary", "INFO", "timing",
                           f"stage {s.index:>2} {s.name:<42} "
                           f"{s.status.upper():<9} {s.duration_seconds:7.2f}s")

    # ----- subprocess helper ---------------------------------------------- #
    def _run_subprocess(self, cmd: List[str], label: str, *,
                        component: str, retries: int) -> Dict[str, Any]:
        def _once() -> Dict[str, Any]:
            self.log.event(component, "INFO", "subprocess_start",
                           f"running {label}", command=" ".join(cmd))
            t0 = time.monotonic()
            completed = subprocess.run(
                cmd, cwd=str(self.cfg.base_dir), env=self._subprocess_env(),
                capture_output=True, text=True,
                timeout=self.cfg.subprocess_timeout, check=False,
            )
            dur = round(time.monotonic() - t0, 2)
            # surface child stderr into our logs (never silent)
            for line in (completed.stderr or "").splitlines()[-40:]:
                if line.strip():
                    self.log.event(component, "DEBUG", "child_stderr", line)
            self.log.event(component, "INFO", "subprocess_end",
                           f"{label} exited rc={completed.returncode}",
                           duration_seconds=dur)
            res = {"returncode": completed.returncode, "stdout": completed.stdout,
                   "stderr": completed.stderr, "duration": dur}
            if completed.returncode != 0 and is_transient_error(
                    RuntimeError(completed.stderr[-300:])):
                raise WorkflowError(f"{label} transient failure rc="
                                    f"{completed.returncode}")
            return res

        try:
            return retry_with_backoff(
                _once, label=label, attempts=max(1, retries + 1),
                base_delay=self.cfg.retry_backoff, logger=self.log,
                component=component, only_transient=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise StageFailure(
                f"{label} timed out after {self.cfg.subprocess_timeout}s") from exc

    # ----- duplicate / chronology helpers --------------------------------- #
    def _sheet_dates(self, ws) -> List[date]:
        values = retry_with_backoff(
            ws.get_all_values, label=f"{ws.title}: dates",
            attempts=self.cfg.google_retries, base_delay=self.cfg.retry_backoff,
            logger=self.log, component="sheets",
        )
        if not values or len(values) < 2:
            return []
        headers = [str(h).strip().lower() for h in values[0]]
        idx = next((i for i, h in enumerate(headers)
                    if h in ("date", "date_str", "datetime")), None)
        if idx is None:
            return []
        out = []
        for row in values[1:]:
            if idx < len(row):
                d = self._parse_date(row[idx])
                if d:
                    out.append(d)
        return out

    def _duplicate_dates(self, ws) -> List[str]:
        dates = self._sheet_dates(ws)
        seen, dups = set(), []
        for d in dates:
            key = d.isoformat()
            if key in seen:
                dups.append(key)
            seen.add(key)
        return dups

    def _is_chronological(self, ws) -> bool:
        dates = self._sheet_dates(ws)
        return all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    # ----- summary --------------------------------------------------------- #
    def build_summary(self) -> Dict[str, Any]:
        failed = [s for s in self.stages if s.status == "failed"]
        degraded = [s for s in self.stages if s.status == "degraded"]
        if failed:
            overall = "error"
        elif self.aborted:
            overall = "aborted"
        elif degraded:
            overall = "success_with_warnings"
        else:
            overall = "success"
        return {
            "run_id": self.run_id,
            "workflow_version": WORKFLOW_VERSION,
            "mode": "live" if self.live else "dry-run",
            "overall_status": overall,
            "aborted": self.aborted,
            "started_at": self.stages[0].started_at if self.stages else utc_iso(),
            "finished_at": utc_iso(),
            "total_duration_seconds": round(time.monotonic() - self.started_monotonic, 2),
            "target_worksheets": self.ctx.get("worksheets", []),
            "test_sheet_id": self.cfg.test_sheet_id,
            "train_sheet_id": self.cfg.train_sheet_id,
            "stages_run": len(self.stages),
            "stages_failed": [s.name for s in failed],
            "stages_degraded": [s.name for s in degraded],
            "log_counts": self.log.counts,
            "logs": {
                "human": str(self.log.human_path),
                "structured_jsonl": str(self.log.jsonl_path),
                "failures": str(self.log.failures_path),
            },
            "stages": [s.to_dict() for s in self.stages],
        }

    # ----- main driver ----------------------------------------------------- #
    def run(self) -> Dict[str, Any]:
        self.log.banner(
            f"STOCK MARKET MASTER WORKFLOW  |  run_id={self.run_id}  |  "
            f"mode={'LIVE' if self.live else 'DRY-RUN'}")
        self.log.event("orchestrator", "INFO", "workflow_start",
                       "workflow started", run_id=self.run_id,
                       mode="live" if self.live else "dry-run",
                       base_dir=str(self.cfg.base_dir),
                       test_sheet=self.cfg.test_sheet_id,
                       train_sheet=self.cfg.train_sheet_id)

        stage_plan: List[Tuple[int, str, Callable[[StageResult], None]]] = [
            (1, "Startup validation", self.stage_01_startup),
            (2, "Check YFinance for new data", self.stage_02_check_yfinance),
            (3, "Download / append new market data", self.stage_03_download_append),
            (4, "Update local workbook / data store", self.stage_04_local_store),
            (5, "Sync new rows into Google Sheets", self.stage_05_sheet_sync),
            (6, "Validate workbook / sheet integrity", self.stage_06_validate_integrity),
            (7, "Run Feature Engineering", self.stage_07_feature_engineering),
            (8, "Validate engineered features", self.stage_08_validate_features),
            (9, "Update metadata", self.stage_09_update_metadata),
            (10, "Run model pipeline (main.py)", self.stage_10_model_pipeline),
            (11, "Generate forecasts / predictions", self.stage_11_generate_forecasts),
            (12, "Push forecasts to Google Sheets", self.stage_12_push_forecasts),
            (13, "Train/test rollover validation", self.stage_13_rollover),
            (14, "Validate final outputs", self.stage_14_validate_final),
        ]
        for index, name, fn in stage_plan:
            if self.aborted:
                skipped = StageResult(index=index, name=name,
                                      gating=index in GATING_STAGES,
                                      status="skipped",
                                      error="skipped — workflow aborted at an "
                                            "earlier gating stage")
                self.stages.append(skipped)
                self.log.event("stage", "WARNING", "stage_skipped",
                               f"Stage {index} ({name}) SKIPPED — "
                               "earlier gating stage failed")
                continue
            self._run_stage(index, name, fn)

        # Stage 15 always runs so a summary is produced even after an abort.
        self._run_stage(15, "Log workflow summary", self.stage_15_summary)

        summary = self.build_summary()
        self.log.banner(
            f"WORKFLOW {summary['overall_status'].upper()}  |  "
            f"run_id={self.run_id}  |  "
            f"{summary['total_duration_seconds']}s")
        self.log.event("orchestrator", "INFO", "workflow_end",
                       f"workflow finished: {summary['overall_status']}",
                       failed=summary["stages_failed"],
                       degraded=summary["stages_degraded"])
        return summary


# =========================================================================== #
# Module-level helpers
# =========================================================================== #
def _parse_last_json(stdout: str) -> Optional[Dict[str, Any]]:
    """Return the last JSON object printed on stdout (matches app_data.py)."""
    for line in reversed((stdout or "").splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_full_workflow.py",
        description="Master end-to-end orchestrator for the Stock Market ML "
                    "pipeline (15 deterministic stages).",
    )
    p.add_argument("--live", action="store_true",
                   help="Perform real Google Sheet writes (yfinance append, "
                        "forecast push, train/test rollover). Default is dry-run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Explicit dry-run (default). No Google Sheet is mutated.")
    p.add_argument("--worksheets", default=None,
                   help="Comma-separated worksheet/stock names. Default: all "
                        "stock worksheets discovered in the operational sheet.")
    p.add_argument("--google-credentials", default=None,
                   help="Path to the service-account JSON. Defaults to "
                        "credentials/Credentials_New.json / env vars.")
    p.add_argument("--run-id", default=None,
                   help="Explicit run id (default: auto-generated).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.live and args.dry_run:
        print(json.dumps({"status": "error",
                          "error": "--live and --dry-run are mutually exclusive"}))
        return 1

    orchestrator: Optional[WorkflowOrchestrator] = None

    def _signal_handler(signum, _frame):
        if orchestrator is not None:
            orchestrator.log.event("orchestrator", "CRITICAL", "signal",
                                   f"received signal {signum} — aborting")
            orchestrator.aborted = True
        raise KeyboardInterrupt()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass

    try:
        orchestrator = WorkflowOrchestrator(args)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error",
                          "error": f"orchestrator init failed: {exc}",
                          "traceback": traceback.format_exc()}))
        return 1

    lock_path = orchestrator.cfg.state_dir / "run_full_workflow.lock"
    try:
        with file_lock(lock_path):
            summary = orchestrator.run()
    except WorkflowError as exc:
        orchestrator.log.event("orchestrator", "CRITICAL", "workflow_error",
                               str(exc))
        summary = orchestrator.build_summary()
        summary["overall_status"] = "error"
        summary["error"] = str(exc)
    except KeyboardInterrupt:
        summary = orchestrator.build_summary()
        summary["overall_status"] = "aborted"
        summary["error"] = "interrupted by signal"
    except Exception as exc:  # noqa: BLE001
        orchestrator.log.event("orchestrator", "CRITICAL", "workflow_crash",
                               f"unhandled crash: {exc}",
                               traceback=traceback.format_exc())
        summary = orchestrator.build_summary()
        summary["overall_status"] = "error"
        summary["error"] = str(exc)
    finally:
        try:
            orchestrator.log.close()
        except Exception:
            pass

    # single machine-readable JSON line on stdout (consumed by run_full_workflow.sh)
    print(json.dumps(summary, separators=(",", ":"), default=str))
    return 0 if summary.get("overall_status") in (
        "success", "success_with_warnings", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
