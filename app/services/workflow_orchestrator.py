from __future__ import annotations

import logging
import sys
import time
from typing import List

from app.config.settings import Settings, get_settings
from app.langchain.chain import build_workflow_chain
from app.langchain.schemas import WorkflowDecision
from app.models.schemas import (
    RouteType,
    RunStatus,
    WorkflowRunRequest,
    WorkflowRunResult,
    model_to_dict,
    utc_now,
)
from app.services.slack import SlackNotifier
from app.services.subprocess_runner import SubprocessRunner
from app.utils.logging import get_logger, log_event


class WorkflowOrchestrator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger(__name__)
        self.chain = build_workflow_chain(self.settings.langchain_mode)
        self.runner = SubprocessRunner(self.settings)
        self.slack = SlackNotifier(self.settings)

    def decide(self, request: WorkflowRunRequest) -> WorkflowDecision:
        decision = self.chain.invoke(request)
        if isinstance(decision, WorkflowDecision):
            return decision
        if hasattr(WorkflowDecision, "model_validate"):
            return WorkflowDecision.model_validate(decision)
        return WorkflowDecision.parse_obj(decision)

    async def run(self, request: WorkflowRunRequest) -> WorkflowRunResult:
        started = time.monotonic()
        started_at = utc_now()
        run_id = request.event.run_id if request.event is not None else f"manual-{int(time.time())}"
        decision = self.decide(request)
        decision_dict = model_to_dict(decision)

        log_event(
            self.logger,
            logging.INFO,
            "workflow_decision",
            "Workflow decision created",
            run_id=run_id,
            decision=decision_dict,
        )

        if not decision.should_run or decision.route == RouteType.SKIP:
            return WorkflowRunResult(
                status=RunStatus.SKIPPED,
                run_id=run_id,
                route=decision.route,
                worksheets=decision.worksheets,
                started_at=started_at,
                finished_at=utc_now(),
                duration_seconds=time.monotonic() - started,
                decision=decision_dict,
            )

        command = self._build_command(decision.worksheets, request.reason, run_id)
        result = await self.runner.run(command, cwd=self.settings.base_dir)
        parsed = result.parsed_json or {}
        result_dict = model_to_dict(result)
        ok = result.ok and parsed.get("status") not in {"error", "failed"}
        workflow_status = RunStatus.SUCCESS if ok else RunStatus.ERROR

        payload = {
            "run_id": run_id,
            "route": decision.route.value,
            "worksheets": ",".join(decision.worksheets) if decision.worksheets else "ALL",
            "returncode": result.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": None if ok else (parsed.get("error") or result.stderr[-1000:]),
        }
        if ok:
            self.slack.notify("Prediction workflow completed", payload, is_failure=False)
        else:
            self.slack.notify("Prediction workflow failed", payload, is_failure=True)

        log_event(
            self.logger,
            logging.INFO if ok else logging.ERROR,
            "workflow_finished",
            "Workflow execution finished",
            run_id=run_id,
            status=workflow_status.value,
            result=parsed,
        )

        return WorkflowRunResult(
            status=workflow_status,
            run_id=run_id,
            route=decision.route,
            worksheets=decision.worksheets,
            started_at=started_at,
            finished_at=utc_now(),
            duration_seconds=time.monotonic() - started,
            decision=decision_dict,
            subprocess_result=result_dict,
            update_result=parsed.get("update_result") if isinstance(parsed, dict) else None,
            prediction_result=parsed.get("prediction_result") if isinstance(parsed, dict) else None,
            error=None if ok else (parsed.get("error") or result.stderr),
        )

    def _build_command(self, worksheets: List[str], reason: str, run_id: str) -> List[str]:
        command = [
            sys.executable,
            "app_data.py",
            "run",
            "--sheet-id",
            self.settings.sheet_id,
            "--google-credentials",
            str(self.settings.google_credentials),
            "--output-dir",
            str(self.settings.output_dir),
            "--model-dir",
            str(self.settings.model_dir),
            "--metadata",
            str(self.settings.metadata_path),
            "--device",
            self.settings.device,
            "--reason",
            reason,
            "--run-id",
            run_id,
        ]
        if worksheets:
            command.extend(["--worksheets", ",".join(worksheets)])
        return command

