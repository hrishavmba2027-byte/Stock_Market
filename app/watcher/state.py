from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from app.config.settings import Settings, get_settings
from app.watcher.diff import build_state_from_snapshots, now_iso
from app.utils.logging import get_logger, log_event


class WorkflowStateManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = self.settings.workflow_state_path
        self.logger = get_logger(__name__)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            backup_path = self.path.with_suffix(f".corrupt.{int(os.path.getmtime(self.path))}.json")
            shutil.move(str(self.path), str(backup_path))
            log_event(
                self.logger,
                logging.ERROR,
                "state_corrupt",
                "State file was corrupt and has been moved aside",
                backup_path=str(backup_path),
            )
            return {}

    def save(self, state: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = now_iso()
        temp_path = self.path.with_suffix(".tmp")
        with open(temp_path, "w") as file:
            json.dump(state, file, indent=2, sort_keys=True)
        os.replace(temp_path, self.path)

    def snapshot_initial(self, snapshots: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        state = build_state_from_snapshots(snapshots)
        self.save(state)
        return state

    def replace_snapshots(
        self,
        snapshots: Dict[str, Dict[str, Any]],
        run_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = self.load() or build_state_from_snapshots({})
        state["worksheets"] = snapshots
        if run_result is not None:
            state["last_run_result"] = run_result
            state["last_successful_run_at"] = now_iso()
        self.save(state)
        return state

