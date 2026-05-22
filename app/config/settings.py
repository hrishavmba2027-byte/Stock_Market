from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is installed in Docker
    def load_dotenv(*args, **kwargs):
        return False


load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[2]
DOCKER_CREDENTIALS = Path("/app/credentials/stock-prices-495408-aa549faac3c5.json")
LOCAL_CREDENTIALS = BASE_DIR / "stock-prices-495408-aa549faac3c5.json"
LOCAL_CREDENTIALS_DIR = BASE_DIR / "credentials" / "stock-prices-495408-aa549faac3c5.json"


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _default_credentials_path() -> Path:
    if os.getenv("GOOGLE_CREDENTIALS"):
        return Path(os.environ["GOOGLE_CREDENTIALS"])
    if DOCKER_CREDENTIALS.exists():
        return DOCKER_CREDENTIALS
    if LOCAL_CREDENTIALS_DIR.exists():
        return LOCAL_CREDENTIALS_DIR
    return LOCAL_CREDENTIALS


class Settings(BaseModel):
    base_dir: Path = BASE_DIR
    sheet_id: str = "1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o"
    google_credentials: Path = _default_credentials_path()
    watcher_poll_seconds: int = 10
    workflow_state_path: Path = BASE_DIR / "workflow_state.json"
    logs_dir: Path = BASE_DIR / "logs"
    app_log_path: Path = BASE_DIR / "logs" / "app.log"
    error_log_path: Path = BASE_DIR / "logs" / "error.log"
    outputs_dir: Path = BASE_DIR / "outputs"
    output_dir: Path = BASE_DIR / "outputs" / "main_inference"
    model_dir: Path = BASE_DIR / "outputs" / "Saved_Models"
    metadata_path: Path = BASE_DIR / "outputs" / "pipeline_metadata.json"
    workbook_path: Path = BASE_DIR / "Data" / "nse_stock_data.xlsx"
    api_base_url: str = "http://api:8000"
    api_timeout_seconds: int = 30
    subprocess_timeout_seconds: int = 1800
    subprocess_retries: int = 2
    google_retries: int = 3
    retry_backoff_seconds: float = 2.0
    update_start_date: str = "2015-01-01"
    update_interval: str = "1d"
    device: str = "auto"
    langchain_mode: str = "deterministic"
    slack_webhook_url: str = ""
    slack_notify_success: bool = False
    slack_notify_failures: bool = True
    # P1 data ingestion + sentiment
    reddit_user_agent: str = "stock-market-automation/0.1"
    fundamentals_path: Path = BASE_DIR / "Data" / "archive" / "fundamentals.parquet"
    news_path: Path = BASE_DIR / "Data" / "archive" / "news.parquet"
    reddit_path: Path = BASE_DIR / "Data" / "archive" / "reddit_posts.parquet"
    x_path: Path = BASE_DIR / "Data" / "archive" / "x_posts.parquet"
    sentiment_path: Path = BASE_DIR / "Data" / "archive" / "sentiment_features.parquet"
    indices_path: Path = BASE_DIR / "Data" / "archive" / "indices.parquet"
    sentiment_model: str = "ProsusAI/finbert"
    sentiment_backfill_days: int = 90
    firestore_project: str = ""
    firestore_fundamentals_collection: str = "fundamentals"

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_env(cls) -> "Settings":
        base_dir = Path(os.getenv("BASE_DIR", str(BASE_DIR)))
        logs_dir = Path(os.getenv("LOGS_DIR", str(base_dir / "logs")))
        outputs_dir = Path(os.getenv("OUTPUTS_DIR", str(base_dir / "outputs")))
        return cls(
            base_dir=base_dir,
            sheet_id=os.getenv("SHEET_ID", "1uekPHyvJj4p6YjxNwlBBIAI71SWRye-xxFu47Kgpf9o"),
            google_credentials=Path(os.getenv("GOOGLE_CREDENTIALS", str(_default_credentials_path()))),
            watcher_poll_seconds=_int_env("WATCHER_POLL_SECONDS", 10),
            workflow_state_path=Path(os.getenv("WORKFLOW_STATE_PATH", str(base_dir / "workflow_state.json"))),
            logs_dir=logs_dir,
            app_log_path=Path(os.getenv("APP_LOG_PATH", str(logs_dir / "app.log"))),
            error_log_path=Path(os.getenv("ERROR_LOG_PATH", str(logs_dir / "error.log"))),
            outputs_dir=outputs_dir,
            output_dir=Path(os.getenv("OUTPUT_DIR", str(outputs_dir / "main_inference"))),
            model_dir=Path(os.getenv("MODEL_DIR", str(outputs_dir / "Saved_Models"))),
            metadata_path=Path(os.getenv("METADATA_PATH", str(outputs_dir / "pipeline_metadata.json"))),
            workbook_path=Path(os.getenv("WORKBOOK", str(base_dir / "Data" / "nse_stock_data.xlsx"))),
            api_base_url=os.getenv("API_BASE_URL", "http://api:8000"),
            api_timeout_seconds=_int_env("API_TIMEOUT_SECONDS", 30),
            subprocess_timeout_seconds=_int_env("SUBPROCESS_TIMEOUT_SECONDS", 1800),
            subprocess_retries=_int_env("SUBPROCESS_RETRIES", 2),
            google_retries=_int_env("GOOGLE_RETRIES", 3),
            retry_backoff_seconds=_float_env("RETRY_BACKOFF_SECONDS", 2.0),
            update_start_date=os.getenv("UPDATE_START_DATE", "2015-01-01"),
            update_interval=os.getenv("UPDATE_INTERVAL", "1d"),
            device=os.getenv("DEVICE", "auto"),
            langchain_mode=os.getenv("LANGCHAIN_MODE", "deterministic"),
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
            slack_notify_success=_bool_env("SLACK_NOTIFY_SUCCESS", False),
            slack_notify_failures=_bool_env("SLACK_NOTIFY_FAILURES", True),
            reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "stock-market-automation/0.1"),
            fundamentals_path=Path(os.getenv("FUNDAMENTALS_PATH", str(base_dir / "Data" / "archive" / "fundamentals.parquet"))),
            news_path=Path(os.getenv("NEWS_PATH", str(base_dir / "Data" / "archive" / "news.parquet"))),
            reddit_path=Path(os.getenv("REDDIT_PATH", str(base_dir / "Data" / "archive" / "reddit_posts.parquet"))),
            x_path=Path(os.getenv("X_PATH", str(base_dir / "Data" / "archive" / "x_posts.parquet"))),
            sentiment_path=Path(os.getenv("SENTIMENT_PATH", str(base_dir / "Data" / "archive" / "sentiment_features.parquet"))),
            indices_path=Path(os.getenv("INDICES_PATH", str(base_dir / "Data" / "archive" / "indices.parquet"))),
            sentiment_model=os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert"),
            sentiment_backfill_days=_int_env("SENTIMENT_BACKFILL_DAYS", 90),
            firestore_project=os.getenv("FIRESTORE_PROJECT", ""),
            firestore_fundamentals_collection=os.getenv("FIRESTORE_FUNDAMENTALS_COLLECTION", "fundamentals"),
        )

    def ensure_runtime_dirs(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workflow_state_path.parent.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> Dict[str, Any]:
        return {
            "sheet_id": self.sheet_id,
            "watcher_poll_seconds": self.watcher_poll_seconds,
            "workflow_state_path": str(self.workflow_state_path),
            "logs_dir": str(self.logs_dir),
            "output_dir": str(self.output_dir),
            "model_dir": str(self.model_dir),
            "metadata_path": str(self.metadata_path),
            "api_base_url": self.api_base_url,
            "subprocess_timeout_seconds": self.subprocess_timeout_seconds,
            "subprocess_retries": self.subprocess_retries,
            "google_retries": self.google_retries,
            "langchain_mode": self.langchain_mode,
            "slack_configured": bool(self.slack_webhook_url),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.ensure_runtime_dirs()
    return settings
