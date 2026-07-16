"""Tiny stage runner shared by the production orchestration scripts.

Each production runbook script (``run_weekly`` / ``run_fortnightly`` /
``run_daily`` / ``run_suggestions``) is just an ordered list of stages, where a
stage is another module run as a subprocess (``python -m ...`` or a top-level
script). Running stages as isolated subprocesses — rather than importing and
calling their ``main()`` — keeps each stage's own CLI, argument parsing and
process-level cleanup intact, and stops one stage's crash from corrupting the
next's interpreter state.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# A stage is (human_name, argv) where argv is passed after the python executable,
# e.g. ("news+sentiment", ["-m", "ingestion.collect_all", "--no-reddit"]).
Stage = Tuple[str, Sequence[str]]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_stage(name: str, argv: Sequence[str], *, dry_run: bool = False) -> Dict[str, Any]:
    """Run one stage as ``python <argv...>``; return a result record."""
    cmd = [sys.executable, *argv]
    printable = " ".join(cmd)
    if dry_run:
        _log(f"[stage] DRY-RUN {name}: {printable}")
        return {"stage": name, "status": "dry_run", "cmd": printable, "returncode": 0}
    _log(f"[stage] ▶ {name}: {printable}")
    started = time.time()
    proc = subprocess.run(cmd)  # inherit stdout/stderr so logs stream live
    elapsed = round(time.time() - started, 1)
    status = "ok" if proc.returncode == 0 else "error"
    _log(f"[stage] {'✓' if status == 'ok' else '✗'} {name} "
         f"({status}, rc={proc.returncode}, {elapsed}s)")
    return {"stage": name, "status": status, "cmd": printable,
            "returncode": proc.returncode, "seconds": elapsed}


def run_pipeline(
    stages: List[Stage],
    *,
    continue_on_error: bool = False,
    dry_run: bool = False,
    title: str = "pipeline",
) -> Dict[str, Any]:
    """Run stages in order. Stops at the first failure unless ``continue_on_error``."""
    _log(f"[{title}] {len(stages)} stages "
         f"({'continue-on-error' if continue_on_error else 'stop-on-error'}"
         f"{', dry-run' if dry_run else ''})")
    results: List[Dict[str, Any]] = []
    for name, argv in stages:
        res = run_stage(name, argv, dry_run=dry_run)
        results.append(res)
        if res["status"] == "error" and not continue_on_error:
            _log(f"[{title}] aborting after failed stage: {name}")
            break
    failed = [r["stage"] for r in results if r["status"] == "error"]
    summary = {
        "pipeline": title,
        "status": "error" if failed else "ok",
        "failed_stages": failed,
        "stages": results,
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary


def add_common_flags(parser) -> None:
    """``--dry-run`` and ``--continue-on-error`` shared by the orchestrators."""
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the stage commands without executing them.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Run every stage even if an earlier one fails.")


def exit_code(summary: Dict[str, Any]) -> int:
    return 0 if summary.get("status") == "ok" else 1
