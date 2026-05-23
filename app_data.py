from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.api.server import app
from app.config.settings import get_settings
from app.pipeline.metadata import initialize_metadata_if_missing
from app.pipeline.startup import run_startup_checks
from app.services.subprocess_runner import parse_last_json
from app.utils.logging import configure_logging, get_logger, log_event


def parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    seen = set()
    result: List[str] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        key = item.upper()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Stock market workflow entrypoint.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run update and prediction workflow.")
    run_parser.add_argument("--sheet-id", default=settings.sheet_id)
    run_parser.add_argument("--google-credentials", default=str(settings.google_credentials))
    run_parser.add_argument("--worksheet", default=None)
    run_parser.add_argument("--worksheets", default=None)
    run_parser.add_argument("--start-date", default=settings.update_start_date)
    run_parser.add_argument("--interval", default=settings.update_interval)
    run_parser.add_argument("--workbook", default=str(settings.workbook_path))
    run_parser.add_argument("--model-dir", default=str(settings.model_dir))
    run_parser.add_argument("--metadata", default=str(settings.metadata_path))
    run_parser.add_argument("--output-dir", default=str(settings.output_dir))
    run_parser.add_argument("--device", default=settings.device, choices=["auto", "cpu", "cuda", "mps"])
    run_parser.add_argument("--reason", default="manual")
    run_parser.add_argument("--run-id", default=f"cli-{int(time.time())}")
    run_parser.add_argument("--plots", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")

    api_parser = subparsers.add_parser("api", help="Run the FastAPI server.")
    api_parser.add_argument("--host", default="0.0.0.0")
    api_parser.add_argument("--port", default=8000, type=int)
    return parser


def command_with_common_args(args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable,
        "Data_update.py",
        "--sheet-id",
        args.sheet_id,
        "--google-credentials",
        args.google_credentials,
        "--start-date",
        args.start_date,
        "--interval",
        args.interval,
    ]
    worksheets = parse_csv(args.worksheets)
    if args.worksheet:
        worksheets.extend(parse_csv(args.worksheet))
    if worksheets:
        command.extend(["--worksheets", ",".join(worksheets)])
    return command


def run_update_subprocess(args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]:
    settings = get_settings()
    command = command_with_common_args(args)
    started = time.monotonic()
    log_event(logger, logging.INFO, "workflow_update_start", "Starting Data_update.py", command=command)
    try:
        completed = subprocess.run(
            command,
            cwd=str(settings.base_dir),
            capture_output=True,
            text=True,
            timeout=settings.subprocess_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "error",
            "error": f"Data_update.py timed out after {settings.subprocess_timeout_seconds} seconds",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "duration_seconds": time.monotonic() - started,
        }

    parsed = parse_last_json(completed.stdout) or {}
    result = {
        "status": parsed.get("status", "ok" if completed.returncode == 0 else "error"),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_seconds": time.monotonic() - started,
        **({"result": parsed} if parsed else {}),
    }
    log_event(
        logger,
        logging.INFO if completed.returncode == 0 else logging.ERROR,
        "workflow_update_finished",
        "Data_update.py finished",
        result={key: value for key, value in result.items() if key not in {"stdout", "stderr"}},
    )
    return result


def run_prediction_pipeline(args: argparse.Namespace, worksheets: List[str], logger: logging.Logger) -> Dict[str, Any]:
    from main import run_pipeline

    started = time.monotonic()
    pipeline_args = Namespace(
        source="google",
        sheet_id=args.sheet_id,
        google_credentials=args.google_credentials,
        worksheet=args.worksheet,
        worksheets=",".join(worksheets) if worksheets else args.worksheets,
        workbook=args.workbook,
        model_dir=args.model_dir,
        metadata=args.metadata,
        output_dir=args.output_dir,
        device=args.device,
        plots=args.plots,
        dry_run=args.dry_run,
    )
    log_event(
        logger,
        logging.INFO,
        "workflow_prediction_start",
        "Starting prediction pipeline",
        worksheets=worksheets,
    )
    result = run_pipeline(pipeline_args)
    result["duration_seconds"] = time.monotonic() - started
    log_event(
        logger,
        logging.INFO if result.get("status") != "error" else logging.ERROR,
        "workflow_prediction_finished",
        "Prediction pipeline finished",
        result=result,
    )
    return result


def _stage(
    logger: logging.Logger,
    event: str,
    message: str,
    **kwargs: Any,
) -> None:
    """Emit a structured stage-transition log entry."""
    log_event(logger, logging.INFO, event, message, **kwargs)


def run_workflow(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Deterministic workflow execution in strict stage order:

        Stage 1 — DATA_UPDATE   : Data_update.py subprocess
        Stage 2 — FE_GATE       : Validate update succeeded before proceeding
        Stage 3 — MODEL_PIPELINE: main.py run_pipeline (FE runs inside here per-symbol)
    """
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger(__name__)
    workflow_started = time.monotonic()
    stage_times: Dict[str, float] = {}

    worksheets = parse_csv(args.worksheets)
    if args.worksheet:
        worksheets.extend(parse_csv(args.worksheet))
    worksheets = parse_csv(",".join(worksheets))

    _stage(
        logger,
        "workflow_start",
        "=== WORKFLOW START ===",
        run_id=args.run_id,
        reason=args.reason,
        worksheets=worksheets,
    )

    try:
        # ── Stage 1: DATA UPDATE ──────────────────────────────────────────────
        _stage(logger, "stage_data_update_begin", "[Stage 1/3] DATA UPDATE — starting Data_update.py")
        t0 = time.monotonic()
        update_result = run_update_subprocess(args, logger)
        stage_times["data_update"] = time.monotonic() - t0

        update_failed = (
            update_result.get("status") == "error"
            and update_result.get("returncode", 1) != 0
        )
        _stage(
            logger,
            "stage_data_update_end",
            "[Stage 1/3] DATA UPDATE — done",
            elapsed=round(stage_times["data_update"], 3),
            returncode=update_result.get("returncode"),
            update_status=update_result.get("status"),
        )

        # ── Stage 2: FE GATE ─────────────────────────────────────────────────
        # Feature Engineering runs inside run_pipeline per-symbol; this gate
        # ensures we never launch the model pipeline after a hard data failure.
        _stage(logger, "stage_fe_gate_begin", "[Stage 2/3] FE GATE — checking data update health")
        if update_failed:
            _stage(
                logger,
                "stage_fe_gate_blocked",
                "[Stage 2/3] FE GATE — BLOCKED: data update hard-failed; "
                "skipping Feature Engineering and Model Pipeline to prevent "
                "inference on stale data",
                error=update_result.get("error") or update_result.get("stderr", "")[:400],
            )
            return {
                "status": "error",
                "run_id": args.run_id,
                "reason": args.reason,
                "worksheets": worksheets,
                "update_result": update_result,
                "error": (
                    "Workflow aborted at FE Gate: Data_update.py exited with "
                    f"returncode={update_result.get('returncode')}. "
                    "Feature Engineering and Model Pipeline were NOT executed."
                ),
                "stage_times": stage_times,
                "duration_seconds": time.monotonic() - workflow_started,
            }

        _stage(
            logger,
            "stage_fe_gate_passed",
            "[Stage 2/3] FE GATE — PASSED: data update healthy; "
            "Feature Engineering will run per-symbol inside Model Pipeline",
        )

        # ── Stage 3: MODEL PIPELINE (FE + inference) ─────────────────────────
        _stage(
            logger,
            "stage_model_pipeline_begin",
            "[Stage 3/3] MODEL PIPELINE — starting (Feature Engineering runs per-symbol here)",
            worksheets=worksheets,
        )
        t0 = time.monotonic()
        prediction_result = run_prediction_pipeline(args, worksheets, logger)
        stage_times["model_pipeline"] = time.monotonic() - t0

        pipeline_status = prediction_result.get("status", "unknown")
        _stage(
            logger,
            "stage_model_pipeline_end",
            "[Stage 3/3] MODEL PIPELINE — done",
            elapsed=round(stage_times["model_pipeline"], 3),
            pipeline_status=pipeline_status,
        )

        status = "success" if pipeline_status != "error" else "error"
        _stage(
            logger,
            "workflow_end",
            f"=== WORKFLOW END — {status.upper()} ===",
            run_id=args.run_id,
            elapsed=round(time.monotonic() - workflow_started, 3),
        )
        return {
            "status": status,
            "run_id": args.run_id,
            "reason": args.reason,
            "worksheets": worksheets,
            "update_result": update_result.get("result", update_result),
            "prediction_result": prediction_result,
            "stage_times": stage_times,
            "duration_seconds": time.monotonic() - workflow_started,
        }

    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "workflow_cli_failed",
            "Workflow CLI failed",
            run_id=args.run_id,
            error=str(exc),
            exc_info=True,
        )
        return {
            "status": "error",
            "run_id": args.run_id,
            "reason": args.reason,
            "worksheets": worksheets,
            "error": str(exc),
            "stage_times": stage_times,
            "duration_seconds": time.monotonic() - workflow_started,
        }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Startup self-checks (non-fatal: logs warnings, never aborts the CLI) ──
    settings = get_settings()
    try:
        # Ensure metadata file exists before any subprocess spawns main.py
        initialize_metadata_if_missing(settings.metadata_path)
        run_startup_checks(settings, strict=False)
    except Exception as exc:
        # Startup checks should never kill the process; surface as a warning
        import logging
        logging.getLogger(__name__).warning(
            "Startup checks raised an exception (non-fatal): %s", exc
        )

    if args.command in {None, "run"}:
        if args.command is None:
            args = parser.parse_args(["run", *sys.argv[1:]])
        result = run_workflow(args)
        print(json.dumps(result, separators=(",", ":"), allow_nan=False))
        raise SystemExit(0 if result.get("status") != "error" else 1)

    if args.command == "api":
        import uvicorn

        uvicorn.run("app_data:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
