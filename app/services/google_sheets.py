from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, TypeVar

from google.oauth2.service_account import Credentials as GoogleCredentials

from app.config.settings import Settings, get_settings
from app.utils.logging import get_logger, log_event


GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

T = TypeVar("T")


class GoogleSheetsService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger(__name__)
        self._client = None

    def client(self):
        if self._client is None:
            import gspread

            creds_path = str(self.settings.google_credentials)
            print(f"[google_auth] Using credentials : {creds_path}", flush=True)
            credentials = GoogleCredentials.from_service_account_file(
                creds_path,
                scopes=GOOGLE_SCOPES,
            )
            print("[google_auth] Service Account Loaded Successfully", flush=True)
            self._client = gspread.authorize(credentials)
            log_event(
                self.logger,
                logging.INFO,
                "google_auth_success",
                "Authorized Google Sheets service account",
                credentials_path=creds_path,
            )
        return self._client

    def spreadsheet(self):
        return self._with_retry(lambda: self.client().open_by_key(self.settings.sheet_id), "open_spreadsheet")

    def worksheets(self) -> List[Any]:
        spreadsheet = self.spreadsheet()
        return self._with_retry(spreadsheet.worksheets, "list_worksheets")

    def get_all_values(self, worksheet: Any) -> List[List[Any]]:
        return self._with_retry(worksheet.get_all_values, "get_all_values", worksheet=getattr(worksheet, "title", ""))

    def _with_retry(self, func: Callable[[], T], operation: str, **context: Any) -> T:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.google_retries + 1):
            try:
                return func()
            except Exception as exc:
                last_error = exc
                log_event(
                    self.logger,
                    logging.WARNING if attempt < self.settings.google_retries else logging.ERROR,
                    "google_operation_failed",
                    "Google Sheets operation failed",
                    operation=operation,
                    attempt=attempt,
                    retries=self.settings.google_retries,
                    error=str(exc),
                    **context,
                )
                if attempt < self.settings.google_retries:
                    time.sleep(self.settings.retry_backoff_seconds * attempt)
        assert last_error is not None
        raise last_error

