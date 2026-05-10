from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

from app.config.settings import Settings, get_settings


_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    RESERVED = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                payload[key] = _json_safe(value)
        if record.exc_info:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(payload, default=str, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def configure_logging(settings: Settings | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = settings or get_settings()
    settings.ensure_runtime_dirs()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = JsonFormatter()
    app_handler = RotatingFileHandler(settings.app_log_path, maxBytes=10_000_000, backupCount=5)
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)

    error_handler = RotatingFileHandler(settings.error_log_path, maxBytes=10_000_000, backupCount=5)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root.addHandler(app_handler)
    root.addHandler(error_handler)
    root.addHandler(stream_handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, event: str, message: str = "", **kwargs: Any) -> None:
    exc_info = kwargs.pop("exc_info", None)
    logger.log(level, message or event, extra={"event": event, **kwargs}, exc_info=exc_info)
