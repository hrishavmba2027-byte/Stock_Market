from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.config.settings import Settings, get_settings
from app.models.schemas import SubprocessResult, model_to_dict
from app.utils.logging import get_logger, log_event


def parse_last_json(stdout: str) -> Optional[Dict]:
    for line in reversed(stdout.splitlines()):
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


class SubprocessRunner:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger(__name__)

    async def run(
        self,
        command: List[str],
        cwd: Path | None = None,
        timeout_seconds: Optional[int] = None,
        retries: Optional[int] = None,
    ) -> SubprocessResult:
        timeout = timeout_seconds or self.settings.subprocess_timeout_seconds
        max_retries = self.settings.subprocess_retries if retries is None else retries
        attempts = max_retries + 1
        last_result: Optional[SubprocessResult] = None

        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            log_event(
                self.logger,
                logging.INFO,
                "subprocess_attempt",
                "Starting subprocess",
                command=command,
                attempt=attempt,
                attempts=attempts,
            )
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(cwd or self.settings.base_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
                duration = time.monotonic() - started
                result = SubprocessResult(
                    command=command,
                    returncode=int(process.returncode or 0),
                    stdout=stdout_bytes.decode(errors="replace"),
                    stderr=stderr_bytes.decode(errors="replace"),
                    parsed_json=parse_last_json(stdout_bytes.decode(errors="replace")),
                    duration_seconds=duration,
                    attempts=attempt,
                    timed_out=False,
                )
            except asyncio.TimeoutError:
                duration = time.monotonic() - started
                result = SubprocessResult(
                    command=command,
                    returncode=124,
                    stdout="",
                    stderr=f"Timed out after {timeout} seconds",
                    parsed_json=None,
                    duration_seconds=duration,
                    attempts=attempt,
                    timed_out=True,
                )

            last_result = result
            log_event(
                self.logger,
                logging.INFO if result.ok else logging.ERROR,
                "subprocess_result",
                "Subprocess finished",
                result=model_to_dict(result),
            )
            if result.ok:
                return result
            if attempt < attempts:
                await asyncio.sleep(self.settings.retry_backoff_seconds * attempt)

        assert last_result is not None
        return last_result

