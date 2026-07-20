"""Backtest configuration — a *separate* config from production ``Settings``.

The backtest is **fully local**: it has no Firestore and no Google Sheet. Prices
come from the local training workbook (``workbook_path``) and news/sentiment from
a local Excel workbook (``news_workbook_path``). Field names may mirror
production, but the *values* are read from ``BACKTEST_*`` env vars so the two
configs never share state.

Semantics that differ from production:

* Backtest news uses the **Wayback CDX** historical scraper (see
  ``backtesting/fetch_news.py``) and aggregates at ``news_interval_days`` (the
  same 7-day cadence as production ``news_lookback_days``), NOT monthly.
* Collected news is **never deleted**: once a ``(target, year)`` pair is in the
  local news workbook it is skipped on re-runs; new years / new stocks append.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parents[2]


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, *, minimum: int | None = None) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _list_env(name: str) -> List[str]:
    """Parse a comma-separated env var into an upper-cased ticker list."""
    raw = os.getenv(name, "") or ""
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _csv_env(name: str) -> List[str]:
    """Parse a comma-separated env var preserving case (for API keys, URLs)."""
    raw = os.getenv(name, "") or ""
    return [t.strip() for t in raw.split(",") if t.strip()]


class BacktestSettings(BaseModel):
    # Master switch. Backtest jobs no-op unless this is on.
    enabled: bool = False

    # ── Simulation window ────────────────────────────────────────────────
    # Inclusive [start_date, end_date]. Empty end_date means "up to today".
    start_date: str = "2020-01-01"
    end_date: str = ""

    # Universe to backtest. Empty list = the full NIFTY universe from aliases.
    tickers: List[str] = []

    # News aggregation / collection cadence in days. Mirrors production
    # news_lookback_days (7): the historical scraper buckets and aggregates news
    # into windows of this many days rather than monthly.
    news_interval_days: int = 7

    # ── Wayback CDX historical news scraper knobs ────────────────────────
    cdx_request_pause_seconds: float = 1.0     # politeness between CDX calls
    cdx_retries: int = 3
    cdx_rate_limit_wait_seconds: float = 15.0  # wait on HTTP 429
    news_content_workers: int = 32             # article-fetch thread pool
    news_fetch_retries: int = 5

    # Local cache/checkpoint dir for the CDX scraper (cache-first + resumable).
    data_dir: Path = BASE_DIR / "backtesting" / "data"

    # Sentiment model (kept separate so backtest can pin a version if needed).
    sentiment_model: str = "ProsusAI/finbert"

    # ── Separate backtest MODEL artifacts (production model is untouched) ──
    # The backtest model is trained on point-in-time data (no look-ahead) and
    # saved here, NOT in outputs/Saved_Models. The walk-forward trainer / the
    # backtest fine-tune entrypoint point monthly_finetune.py at these paths.
    model_dir: Path = BASE_DIR / "outputs" / "backtest" / "Saved_Models"
    metadata_path: Path = BASE_DIR / "outputs" / "backtest" / "pipeline_metadata.json"
    finetune_state_file: Path = BASE_DIR / "state" / "backtest_finetune_state.json"
    output_dir: Path = BASE_DIR / "outputs" / "backtest" / "monthly_finetune"

    # ── Phase-4 walk-forward harness (additive; all env-driven, defaulted so
    #    the existing backtest-config tests keep passing) ──────────────────
    #
    # Portfolio / capital deployment. ``initial_capital`` seeds the simulated
    # book at ₹1 crore; the optimizer keeps a cash reserve (``cash_floor_frac``)
    # so capital is only ever *partially* deployed and can never go negative.
    initial_capital: float = 1_00_00_000.0
    retrain_interval_days: int = 15          # walk-forward retrain/forecast cadence
    sentiment_interval_days: int = 7         # sentiment aggregation window (days of news per snapshot)
    # Weekly open-position review cadence, in TRADING bars (5 ≈ one calendar
    # week). Between retrains, ONLY held names are re-underwritten on these bars
    # (fresh sentiment + latest forecasts through the position-review prompt);
    # fresh entries are decided on retrain bars alone.
    review_interval_days: int = 5
    forecast_days: int = 30                  # T+1..T+30 forecast path (FORECAST_DAYS);
                                             # sizing still uses T+15, the LLM sees the full path
    cash_floor_frac: float = 0.10            # >=10% of equity always retained as cash
    per_name_cap_frac: float = 0.20          # <=20% of the book in any one name
    min_weight_frac: float = 0.05            # drop any position sized below 5% of the book to 0
    vol_target_annual: float = 0.20          # per-name vol-scaled cap target
    llm_confidence_tilt: float = 0.25        # mild expected-return tilt from LLM confidence
    min_ticket_inr: float = 10_000.0         # skip orders smaller than this notional
    # Risk-reward edge (fractional) a *new* BUY must beat an existing holding by
    # before capital is pulled out of the holding to fund it (cash-exhaustion
    # reallocation). 0.15 ⇒ the candidate's reward/risk must exceed the incumbent's
    # by 15% before we rotate capital.
    reallocation_edge: float = 0.15

    # Conviction-scaled deployment: how MUCH of the book to deploy scales with the
    # BUY set's weighted risk-reward. Mediocre deals deploy only ``deploy_floor_frac``
    # (keep cash at hand); deployment reaches 100% (exhaust the cash) only when the
    # weighted risk-reward hits ``rr_full_deploy``.
    deploy_floor_frac: float = 0.30
    rr_min_deploy: float = 1.0
    rr_full_deploy: float = 3.0

    # Target/stop profit-booking: every bar, a held name whose LLM target
    # (``sell_price``) is touched is sold same-bar and the cash held until a fresh
    # suggestion. ``honor_stoploss`` also arms the protective stop (kept on so the
    # backtest is symmetric — it books losers as well as winners, not just wins).
    honor_stoploss: bool = True

    # ── Entry gates (trade-quality findings, 2026-07) ───────────────────────
    # Deterministic gates applied to every LLM BUY; see features/entry_gates.py
    # for the numbers behind each default. momentum_lookback_days doubles as the
    # historical lookback AND the forecast horizon of the agreement gate — 15
    # maximized win rate and per-trade P&L on the 2020–2026 grid.
    entry_gates_enabled: bool = True
    momentum_lookback_days: int = 15
    gate_rsi_min: float = 45.0
    gate_rsi_max: float = 70.0
    gate_volume_spike_ratio: float = 1.5
    gate_stop_atr_mult: float = 2.0
    # Cost-adjusted RR floor. 1.5 was calibrated for the old sell-at-target
    # world; with hold-and-trail the upside is no longer capped at the stated
    # target, so 1.0 (reject only inverted geometry) is the 2026-07-20 default —
    # the 1.5 bar blocked 50/76 BUY intents incl. all of 2021 (6/8 profitable).
    gate_min_eff_rr: float = 1.0
    gate_roundtrip_cost_pct: float = 0.009
    gate_max_chase_pct: float = 0.03
    # Judicious negative-forecast override (default: never fires — every
    # historical slice of it lost money; raise via env only deliberately).
    gate_neg_fc_override_min_confidence: float = 1.01
    gate_neg_fc_override_min_momentum: float = 0.06
    # Time-diversification: at most this many NEW positions may be opened per
    # rebalance cycle (0 = unlimited). Overflow BUYs (ranked by cost-adjusted
    # RR, then confidence) are deferred and re-qualify at the next cycle with
    # fresh forecasts — 5 of the 6 2022 entries landed in one 5-day window and
    # fell together (−₹585k); staggering caps that single-week concentration.
    max_new_positions_per_cycle: int = 3

    # ── Hold-and-trail exits (findings D/E) ─────────────────────────────────
    # On target touch, do NOT auto-sell: ratchet the stop to
    # max(prev stop, close − trail_atr_mult*ATR14, entry) and keep the position
    # until the stop is hit, the LLM verdict changes, or max_holding_days ages
    # it out. The 16–30 day bucket was the only profitable one (61% win rate).
    trail_on_target: bool = True
    trail_atr_mult: float = 2.0
    # Age-based exit DISABLED (0) per user decision 2026-07-20: exits are driven
    # by the stop, the trail, and LLM verdict changes only — a position with an
    # intact thesis is not closed merely for being old.
    max_holding_days: int = 0                # calendar days; 0 disables the age exit
    # Profit cushion a position must clear before its stop may be tightened to
    # break-even or above. Prevents the "raised to entry after +0.9% then wicked
    # out flat" whipsaw (ULTRACEMCO 2021-02). Arm level =
    # entry + max(trail_arm_atr_mult*ATR14, trail_arm_profit_pct*entry).
    trail_arm_profit_pct: float = 0.02
    trail_arm_atr_mult: float = 1.0

    # Execution frictions.
    cost_bps_roundtrip: float = 40.0         # NSE brokerage+STT+txn+GST+stamp (bps, split/leg)
    slippage_atr_mult: float = 0.10          # slippage = mult * ATR14 per share

    # Benchmark + fundamentals PIT gate.
    benchmark_symbol: str = "^NSEI"
    reporting_lag_days: int = 45             # yfinance-fallback fundamentals visibility lag

    # Cold-start seed training (fresh weights on <= first-cutoff data).
    seed_epochs: int = 60

    # ── Rotating LLM key pool (backtest only) ─────────────────────────────
    # The long walk-forward exhausts a single key's rate/token budget, so the
    # backtest rotates through a pool: all Ollama Cloud keys first, then the
    # Groq keys (both serving gpt-oss-120b), wrapping back to Ollama key 1 after
    # the last Groq key. On ANY provider error the client advances to the next
    # key and continues. Populated from BACKTEST_OLLAMA_API_KEYS /
    # BACKTEST_GROQ_API_KEYS (comma-separated). Empty → the harness falls back to
    # the single GLM_API_KEY / GLM_MODEL used by production.
    ollama_api_keys: List[str] = []
    groq_api_keys: List[str] = []
    ollama_base_url: str = "https://ollama.com/v1"
    ollama_model: str = "gpt-oss:120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"

    # Local bookkeeping workbook (all backtest LLM suggestions, allocations,
    # fills, sales and the running P&L land here — one row per event).
    bookkeeping_path: Path = BASE_DIR / "outputs" / "backtest" / "backtest_bookkeeping.xlsx"
    state_path: Path = BASE_DIR / "outputs" / "backtest" / "state" / "backtest_state.json"
    report_dir: Path = BASE_DIR / "outputs" / "backtest" / "report"
    workbook_path: Path = BASE_DIR / "Data" / "archive" / "nse_stock_data_train_filled.xlsx"
    index_cache_path: Path = BASE_DIR / "Data" / "archive" / "indices.parquet"
    sentiment_cache_path: Path = BASE_DIR / "outputs" / "backtest" / "sentiment_cache.parquet"
    # Local news/sentiment workbook — the single, fully-local source the backtest
    # reads (no Firestore, no Google Sheet). Sheets: ``News`` (raw scraped
    # articles), ``Sentiment`` (per-window aggregates the LLM consumes), and
    # ``Manifest`` ((target, year) collection checkpoints).
    news_workbook_path: Path = BASE_DIR / "Data" / "archive" / "backtest_news.xlsx"

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_env(cls) -> "BacktestSettings":
        base_dir = Path(os.getenv("BASE_DIR", str(BASE_DIR)))
        return cls(
            enabled=_bool_env("BACKTEST_ENABLED", False),
            start_date=os.getenv("BACKTEST_START_DATE", "2020-01-01"),
            end_date=os.getenv("BACKTEST_END_DATE", ""),
            tickers=_list_env("BACKTEST_TICKERS"),
            news_interval_days=_int_env("BACKTEST_NEWS_INTERVAL_DAYS", 7, minimum=1),
            cdx_request_pause_seconds=_float_env("BACKTEST_CDX_PAUSE_SECONDS", 1.0),
            cdx_retries=_int_env("BACKTEST_CDX_RETRIES", 3, minimum=0),
            cdx_rate_limit_wait_seconds=_float_env("BACKTEST_CDX_RATE_LIMIT_WAIT_SECONDS", 15.0),
            news_content_workers=_int_env("BACKTEST_NEWS_CONTENT_WORKERS", 32, minimum=1),
            news_fetch_retries=_int_env("BACKTEST_NEWS_FETCH_RETRIES", 5, minimum=0),
            data_dir=Path(os.getenv("BACKTEST_DATA_DIR", str(base_dir / "backtesting" / "data"))),
            sentiment_model=os.getenv("BACKTEST_SENTIMENT_MODEL", "ProsusAI/finbert"),
            model_dir=Path(os.getenv("BACKTEST_MODEL_DIR", str(base_dir / "outputs" / "backtest" / "Saved_Models"))),
            metadata_path=Path(os.getenv("BACKTEST_METADATA_PATH", str(base_dir / "outputs" / "backtest" / "pipeline_metadata.json"))),
            finetune_state_file=Path(os.getenv("BACKTEST_FINETUNE_STATE_FILE", str(base_dir / "state" / "backtest_finetune_state.json"))),
            output_dir=Path(os.getenv("BACKTEST_OUTPUT_DIR", str(base_dir / "outputs" / "backtest" / "monthly_finetune"))),
            initial_capital=_float_env("BACKTEST_INITIAL_CAPITAL", 1_00_00_000.0),
            retrain_interval_days=_int_env("BACKTEST_RETRAIN_INTERVAL_DAYS", 15, minimum=1),
            sentiment_interval_days=_int_env("BACKTEST_SENTIMENT_INTERVAL_DAYS", 7, minimum=1),
            review_interval_days=_int_env("BACKTEST_REVIEW_INTERVAL_DAYS", 5, minimum=1),
            forecast_days=_int_env("BACKTEST_FORECAST_DAYS", 30, minimum=1),
            cash_floor_frac=_float_env("BACKTEST_CASH_FLOOR_FRAC", 0.10),
            per_name_cap_frac=_float_env("BACKTEST_PER_NAME_CAP_FRAC", 0.20),
            min_weight_frac=_float_env("BACKTEST_MIN_WEIGHT_FRAC", 0.05),
            vol_target_annual=_float_env("BACKTEST_VOL_TARGET_ANNUAL", 0.20),
            llm_confidence_tilt=_float_env("BACKTEST_LLM_CONFIDENCE_TILT", 0.25),
            min_ticket_inr=_float_env("BACKTEST_MIN_TICKET_INR", 10_000.0),
            reallocation_edge=_float_env("BACKTEST_REALLOCATION_EDGE", 0.15),
            deploy_floor_frac=_float_env("BACKTEST_DEPLOY_FLOOR_FRAC", 0.30),
            rr_min_deploy=_float_env("BACKTEST_RR_MIN_DEPLOY", 1.0),
            rr_full_deploy=_float_env("BACKTEST_RR_FULL_DEPLOY", 3.0),
            honor_stoploss=_bool_env("BACKTEST_HONOR_STOPLOSS", True),
            entry_gates_enabled=_bool_env("BACKTEST_ENTRY_GATES_ENABLED", True),
            momentum_lookback_days=_int_env("BACKTEST_MOMENTUM_LOOKBACK_DAYS", 15, minimum=1),
            gate_rsi_min=_float_env("BACKTEST_GATE_RSI_MIN", 45.0),
            gate_rsi_max=_float_env("BACKTEST_GATE_RSI_MAX", 70.0),
            gate_volume_spike_ratio=_float_env("BACKTEST_GATE_VOLUME_SPIKE_RATIO", 1.5),
            gate_stop_atr_mult=_float_env("BACKTEST_GATE_STOP_ATR_MULT", 2.0),
            gate_min_eff_rr=_float_env("BACKTEST_GATE_MIN_EFF_RR", 1.0),
            gate_roundtrip_cost_pct=_float_env("BACKTEST_GATE_ROUNDTRIP_COST_PCT", 0.009),
            gate_max_chase_pct=_float_env("BACKTEST_GATE_MAX_CHASE_PCT", 0.03),
            gate_neg_fc_override_min_confidence=_float_env("BACKTEST_GATE_NEG_FC_OVERRIDE_MIN_CONFIDENCE", 1.01),
            gate_neg_fc_override_min_momentum=_float_env("BACKTEST_GATE_NEG_FC_OVERRIDE_MIN_MOMENTUM", 0.06),
            max_new_positions_per_cycle=_int_env("BACKTEST_MAX_NEW_POSITIONS_PER_CYCLE", 3, minimum=0),
            trail_on_target=_bool_env("BACKTEST_TRAIL_ON_TARGET", True),
            trail_atr_mult=_float_env("BACKTEST_TRAIL_ATR_MULT", 2.0),
            max_holding_days=_int_env("BACKTEST_MAX_HOLDING_DAYS", 0, minimum=0),
            trail_arm_profit_pct=_float_env("BACKTEST_TRAIL_ARM_PROFIT_PCT", 0.02),
            trail_arm_atr_mult=_float_env("BACKTEST_TRAIL_ARM_ATR_MULT", 1.0),
            cost_bps_roundtrip=_float_env("BACKTEST_COST_BPS_ROUNDTRIP", 40.0),
            slippage_atr_mult=_float_env("BACKTEST_SLIPPAGE_ATR_MULT", 0.10),
            benchmark_symbol=os.getenv("BACKTEST_BENCHMARK_SYMBOL", "^NSEI"),
            reporting_lag_days=_int_env("BACKTEST_REPORTING_LAG_DAYS", 45, minimum=0),
            seed_epochs=_int_env("BACKTEST_SEED_EPOCHS", 60, minimum=1),
            bookkeeping_path=Path(os.getenv("BACKTEST_BOOKKEEPING_PATH", str(base_dir / "outputs" / "backtest" / "backtest_bookkeeping.xlsx"))),
            state_path=Path(os.getenv("BACKTEST_STATE_PATH", str(base_dir / "outputs" / "backtest" / "state" / "backtest_state.json"))),
            report_dir=Path(os.getenv("BACKTEST_REPORT_DIR", str(base_dir / "outputs" / "backtest" / "report"))),
            workbook_path=Path(os.getenv("BACKTEST_WORKBOOK_PATH", str(base_dir / "Data" / "archive" / "nse_stock_data_train_filled.xlsx"))),
            index_cache_path=Path(os.getenv("BACKTEST_INDEX_CACHE_PATH", str(base_dir / "Data" / "archive" / "indices.parquet"))),
            sentiment_cache_path=Path(os.getenv("BACKTEST_SENTIMENT_CACHE_PATH", str(base_dir / "outputs" / "backtest" / "sentiment_cache.parquet"))),
            news_workbook_path=Path(os.getenv("BACKTEST_NEWS_WORKBOOK_PATH", str(base_dir / "Data" / "archive" / "backtest_news.xlsx"))),
            ollama_api_keys=_csv_env("BACKTEST_OLLAMA_API_KEYS"),
            groq_api_keys=_csv_env("BACKTEST_GROQ_API_KEYS"),
            ollama_base_url=os.getenv("BACKTEST_OLLAMA_BASE_URL", "https://ollama.com/v1"),
            ollama_model=os.getenv("BACKTEST_OLLAMA_MODEL", "gpt-oss:120b"),
            groq_base_url=os.getenv("BACKTEST_GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            groq_model=os.getenv("BACKTEST_GROQ_MODEL", "openai/gpt-oss-120b"),
        )

    def resolved_end_date(self) -> str:
        """End date, defaulting to today (UTC) when unset."""
        if self.end_date.strip():
            return self.end_date.strip()
        from datetime import date

        return date.today().isoformat()

    def news_years(self) -> List[int]:
        """Calendar years spanned by [start_date, resolved_end_date]."""
        start_year = int(self.start_date[:4])
        end_year = int(self.resolved_end_date()[:4])
        return list(range(start_year, end_year + 1))

    def resolved_tickers(self) -> List[str]:
        """Backtest universe — configured subset, or the full NIFTY universe."""
        if self.tickers:
            return list(self.tickers)
        from ingestion.aliases import list_tickers

        return list_tickers()

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "start_date": self.start_date,
            "end_date": self.resolved_end_date(),
            "tickers": self.tickers or "ALL",
            "news_interval_days": self.news_interval_days,
            "news_workbook": str(self.news_workbook_path),
            "data_dir": str(self.data_dir),
        }


@lru_cache(maxsize=1)
def get_backtest_settings() -> BacktestSettings:
    return BacktestSettings.from_env()
