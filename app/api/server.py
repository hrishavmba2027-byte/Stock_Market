from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import Body, FastAPI
from fastapi.encoders import jsonable_encoder

from app.config.settings import get_settings
from app.models.schemas import WorkflowRunRequest, WorkflowStatus, model_to_dict
from app.services.workflow_orchestrator import WorkflowOrchestrator
from app.utils.logging import configure_logging


settings = get_settings()
configure_logging(settings)
app = FastAPI(title="Stock Market Automation", version="1.0.0")
orchestrator = WorkflowOrchestrator(settings)
run_lock = asyncio.Lock()
LAST_RUN = None


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "stock-market-automation",
        "state_exists": settings.workflow_state_path.exists(),
    }


@app.get("/status")
async def status():
    payload = WorkflowStatus(
        last_run=LAST_RUN,
        state_exists=settings.workflow_state_path.exists(),
        config=settings.public_dict(),
    )
    return jsonable_encoder(model_to_dict(payload))


@app.post("/run")
async def run_workflow(payload: Optional[WorkflowRunRequest] = Body(default=None)):
    global LAST_RUN
    request = payload or WorkflowRunRequest(reason="api")
    async with run_lock:
        result = await orchestrator.run(request)
        LAST_RUN = model_to_dict(result)
        return jsonable_encoder(LAST_RUN)


@app.get("/run")
async def run_compat(worksheet: Optional[str] = None, force: bool = True):
    worksheets = [worksheet] if worksheet else []
    request = WorkflowRunRequest(worksheets=worksheets, force=force, reason="compat_get_run")
    return await run_workflow(request)

