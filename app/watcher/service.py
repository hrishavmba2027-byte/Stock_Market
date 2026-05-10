from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

import requests

from app.config.settings import Settings, get_settings
from app.models.schemas import SheetChangeEvent, WorkflowRunRequest, model_to_dict
from app.services.google_sheets import GoogleSheetsService
from app.utils.logging import configure_logging, get_logger, log_event
from app.watcher.diff import build_worksheet_snapshot, diff_snapshots, has_actionable_changes
from app.watcher.state import WorkflowStateManager


class SheetsWatcher:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        configure_logging(self.settings)
        self.logger = get_logger(__name__)
        self.google = GoogleSheetsService(self.settings)
        self.state = WorkflowStateManager(self.settings)

    def read_snapshots(self) -> Dict[str, Dict[str, Any]]:
        snapshots: Dict[str, Dict[str, Any]] = {}
        for worksheet in self.google.worksheets():
            try:
                values = self.google.get_all_values(worksheet)
                snapshot = build_worksheet_snapshot(worksheet, values)
                snapshots[snapshot["worksheet_id"]] = snapshot
            except Exception as exc:
                log_event(
                    self.logger,
                    logging.ERROR,
                    "worksheet_snapshot_failed",
                    "Failed to snapshot worksheet",
                    worksheet=getattr(worksheet, "title", ""),
                    error=str(exc),
                )
        return snapshots

    async def run_forever(self) -> None:
        log_event(
            self.logger,
            logging.INFO,
            "watcher_started",
            "Google Sheets watcher started",
            poll_seconds=self.settings.watcher_poll_seconds,
        )
        while True:
            try:
                await self.poll_once()
            except Exception as exc:
                log_event(
                    self.logger,
                    logging.ERROR,
                    "watcher_poll_failed",
                    "Watcher poll failed",
                    error=str(exc),
                    exc_info=True,
                )
            await asyncio.sleep(self.settings.watcher_poll_seconds)

    async def poll_once(self) -> None:
        current_snapshots = self.read_snapshots()
        state = self.state.load()
        if not state or not state.get("worksheets"):
            self.state.snapshot_initial(current_snapshots)
            log_event(
                self.logger,
                logging.INFO,
                "watcher_initial_snapshot",
                "Initial worksheet snapshot created without triggering workflow",
                worksheet_count=len(current_snapshots),
            )
            return

        changes = diff_snapshots(state, current_snapshots)
        if not changes:
            log_event(self.logger, logging.INFO, "watcher_no_changes", "No worksheet changes detected")
            return

        if not has_actionable_changes(changes):
            self.state.replace_snapshots(current_snapshots)
            log_event(
                self.logger,
                logging.INFO,
                "watcher_non_actionable_changes",
                "Only non-actionable worksheet changes detected",
                changes=[model_to_dict(change) for change in changes],
            )
            return

        event = SheetChangeEvent(spreadsheet_id=self.settings.sheet_id, changes=changes)
        result = await self.call_api(event)
        if result and result.get("status") in {"success", "skipped"}:
            refreshed = self.read_snapshots()
            self.state.replace_snapshots(refreshed, run_result=result)
            log_event(
                self.logger,
                logging.INFO,
                "watcher_state_refreshed",
                "State refreshed after successful workflow run",
                run_id=event.run_id,
            )
        else:
            log_event(
                self.logger,
                logging.ERROR,
                "watcher_workflow_failed",
                "Workflow failed; keeping previous state so changes can retry",
                run_id=event.run_id,
                result=result,
            )

    async def call_api(self, event: SheetChangeEvent) -> Dict[str, Any]:
        request = WorkflowRunRequest(event=event, reason="watcher_change")
        payload = model_to_dict(request)
        url = f"{self.settings.api_base_url.rstrip('/')}/run"
        last_error = None
        for attempt in range(1, self.settings.google_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.settings.api_timeout_seconds)
                response.raise_for_status()
                result = response.json()
                log_event(
                    self.logger,
                    logging.INFO,
                    "watcher_api_call_success",
                    "Watcher API call succeeded",
                    run_id=event.run_id,
                    result=result,
                )
                return result
            except Exception as exc:
                last_error = exc
                log_event(
                    self.logger,
                    logging.ERROR if attempt == self.settings.google_retries else logging.WARNING,
                    "watcher_api_call_failed",
                    "Watcher API call failed",
                    run_id=event.run_id,
                    attempt=attempt,
                    error=str(exc),
                )
                await asyncio.sleep(self.settings.retry_backoff_seconds * attempt)
        return {"status": "error", "error": str(last_error)}


async def async_main() -> None:
    watcher = SheetsWatcher()
    await watcher.run_forever()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

