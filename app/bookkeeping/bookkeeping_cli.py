"""Command-line / JSON interface to the bookkeeping engine.

This is the wrapper Claude Desktop calls. It is deliberately thin: it parses
arguments, hands a JSON trade request to :class:`BookkeepingEngine`, and prints
the structured JSON response on stdout.

Exit codes
----------
* ``0`` -- the engine produced a result (whether the trade was approved or
  rejected -- a rejection is a valid, expected outcome).
* ``2`` -- a *system* failure (bad JSON input, storage unreachable, etc.).

Usage
-----
Provision the backend (creates the worksheets)::

    python -m app.bookkeeping.bookkeeping_cli init

Inspect connectivity / current ledger state::

    python -m app.bookkeeping.bookkeeping_cli health
    python -m app.bookkeeping.bookkeeping_cli state

Evaluate a trade request (Claude's main call) -- request JSON may come from an
argument, a file, or stdin::

    python -m app.bookkeeping.bookkeeping_cli process --json '{"trade_mode":"single","symbol":"RELIANCE","quantity":10,"price":2450.5,"request_id":"abc123"}'
    python -m app.bookkeeping.bookkeeping_cli process --file request.json
    echo '{...}' | python -m app.bookkeeping.bookkeeping_cli process

Add ``--dry-run`` to evaluate without committing (logs an open suggestion).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict

from app.bookkeeping.bookkeeping import BookkeepingEngine

EXIT_OK = 0
EXIT_SYSTEM_ERROR = 2


def _print(payload: Dict[str, Any]) -> None:
    """Emit a single JSON object on stdout (the contract Claude reads)."""
    sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
    sys.stdout.flush()


def _read_request(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve the trade request JSON from --json, --file, or stdin."""
    if args.json:
        raw = args.json
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as handle:
            raw = handle.read()
    else:
        if sys.stdin.isatty():
            raise ValueError(
                "No trade request supplied. Use --json, --file, or pipe JSON via stdin."
            )
        raw = sys.stdin.read()

    if not raw or not raw.strip():
        raise ValueError("Trade request input is empty.")
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Trade request is not valid JSON: {exc}") from exc
    if not isinstance(request, dict):
        raise ValueError("Trade request JSON must be an object.")
    return request


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bookkeeping_cli",
        description="Trade-capital control / bookkeeping engine.",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        help="Python logging level for stderr diagnostics (default: WARNING).",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Create/repair the storage backend schema.")
    sub.add_parser("health", help="Report backend connectivity and config.")
    sub.add_parser("state", help="Print the current ledger state snapshot.")

    process = sub.add_parser(
        "process", help="Evaluate (and, unless --dry-run, execute) a trade request."
    )
    process.add_argument("--json", help="Trade request as an inline JSON string.")
    process.add_argument("--file", help="Path to a JSON file with the trade request.")
    process.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate only; log an open suggestion and do not commit capital.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Default command: 'process' (so piping JSON with no subcommand works).
    command = args.command or "process"

    try:
        if command == "init":
            engine = BookkeepingEngine()
            ok, detail = engine.init_backend()
            _print({"command": "init", "ok": ok, "detail": detail})
            return EXIT_OK if ok else EXIT_SYSTEM_ERROR

        if command == "health":
            engine = BookkeepingEngine()
            _print({"command": "health", **engine.health_check()})
            return EXIT_OK

        if command == "state":
            engine = BookkeepingEngine()
            _print({"command": "state", **engine.get_state_summary()})
            return EXIT_OK

        if command == "process":
            request = _read_request(args)
            if getattr(args, "dry_run", False):
                request["dry_run"] = True
            engine = BookkeepingEngine()
            response = engine.process(request)
            _print(response)
            return EXIT_OK

        parser.print_help(sys.stderr)
        return EXIT_SYSTEM_ERROR

    except ValueError as exc:
        # Bad input -- report as a structured error, not a stack trace.
        _print({"can_trade": False, "error": str(exc), "ledger_update_status": "skipped"})
        return EXIT_SYSTEM_ERROR
    except Exception as exc:  # noqa: BLE001 - top-level guard
        logging.getLogger("bookkeeping.cli").exception("Unhandled CLI failure")
        _print({
            "can_trade": False,
            "error": f"System error: {exc}",
            "ledger_update_status": "error",
        })
        return EXIT_SYSTEM_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
