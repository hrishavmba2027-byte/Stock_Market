from __future__ import annotations

import logging
from typing import Any, Dict

import requests

from app.config.settings import Settings, get_settings
from app.utils.logging import get_logger, log_event


class SlackNotifier:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger(__name__)

    def notify(self, title: str, payload: Dict[str, Any], is_failure: bool = True) -> bool:
        if is_failure and not self.settings.slack_notify_failures:
            return False
        if not is_failure and not self.settings.slack_notify_success:
            return False
        if not self.settings.slack_webhook_url:
            log_event(self.logger, logging.INFO, "slack_noop", "Slack webhook not configured", title=title)
            return False

        text = self._format_message(title, payload, is_failure)
        try:
            response = requests.post(
                self.settings.slack_webhook_url,
                json={"text": text},
                timeout=10,
            )
            response.raise_for_status()
            log_event(self.logger, logging.INFO, "slack_sent", "Slack notification sent", title=title)
            return True
        except Exception as exc:
            log_event(
                self.logger,
                logging.ERROR,
                "slack_failed",
                "Slack notification failed",
                title=title,
                error=str(exc),
            )
            return False

    @staticmethod
    def _format_message(title: str, payload: Dict[str, Any], is_failure: bool) -> str:
        status = "FAILED" if is_failure else "COMPLETED"
        details = "\n".join(f"*{key}:* {value}" for key, value in payload.items() if value is not None)
        return f"Stock Prediction {status}: {title}\n{details}"

