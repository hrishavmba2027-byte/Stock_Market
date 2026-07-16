# Backtesting harness — how to run it

A walk-forward simulation that replays the **exact production pipeline** over a
historical window (default **2020-01-01 → 2026-06-30**) for the 20-stock backtest
universe, then benchmarks the result against NIFTY 50, Sensex and government
bonds. Everything is **fully local** (no Firestore, no Google Sheets) and strictly
**point-in-time** — at every decision date `C`, prices/technicals, news sentiment,
fundamentals and the model are all cut at `C`; the only forward-looking quantity
is the model's own T+1..T+15 forecast.

There are three stages, run in order:

```
1. COLLECT   news + fundamentals for the window      → local files
2. SIMULATE  walk-forward replay + bookkeeping        → state + bookkeeping.xlsx
3. RESULTS   benchmark sheet vs NIFTY/Sensex/bonds     → results.xlsx + chart
```

---

## 0. One-time setup

```bash
# Backtest universe + master switch (news collection needs BACKTEST_ENABLED)
export BACKTEST_ENABLED=true
export BACKTEST_TICKERS="HDFCBANK,ICICIBANK,KOTAKBANK,SBIN,TCS,INFY,RELIANCE,ONGC,ITC,HINDUNILVR,MARUTI,M&M,SUNPHARMA,CIPLA,TATASTEEL,HINDALCO,NTPC,BHARTIARTL,LT,ULTRACEMCO"
export BACKTEST_START_DATE=2020-01-01
export BACKTEST_END_DATE=2026-06-30

# LLM (BUY/HOLD/AVOID) — the production decision layer
export GLM_API_KEY=...            # required for the simulation
# export GLM_MODEL=glm-4.7        # optional override

# Optional: only if screener.in rate-limits anonymous requests
# export SCREENER_COOKIE="sessionid=...; csrftoken=..."
```

(These already live in `.env`; export them or rely on `.env` auto-loading.)

Required inputs already in the repo:
- **Prices**: `Data/archive/nse_stock_data_train_filled.xlsx` (`BACKTEST_WORKBOOK_PATH`).
- **Index benchmark cache** (optional but recommended for the per-bar NIFTY line in
  the sim report): build once —
  ```bash
  python -m features.cross_sectional --output Data/archive/indices.parquet --start 2019-01-01
  ```
  If absent, the simulation still runs; the `results` stage fetches NIFTY/Sensex
  live via yfinance anyway.

All config knobs are `BACKTEST_*` env vars in
[`app/config/backtest_settings.py`](../app/config/backtest_settings.py).

---

## 1. Collect data (news + fundamentals)

**Common collector — run this once before simulating:**

```bash
python -m backtesting.collect_data
```

This runs both collectors for the universe/window:

- **Fundamentals — [`backtesting/fetch_fundamentals.py`](fetch_fundamentals.py)** (source: **screener.in**).
  Scrapes each company's quarterly-results table (~12 quarters) **and** 10-year
  annual P&L, and appends them to the fundamentals **PIT parquet**
  (`Data/pit/fundamentals.parquet`). Each period is stamped with its
  *information-availability date* in the `scrape_date` column, so a cutoff `C` only
  ever sees results already announced by `C`:
  - **Quarterly** → visible `quarter_end + ~45d` (the results-announcement lag).
  - **Annual** → visible in **June** of the year the fiscal year closed
    (`--annual-refresh-month`, default 6). So a Jan/Feb 2020 cutoff sees FY2019; a
    June/July 2020 cutoff sees FY2020.

  Idempotent (dedup on `ticker,quarter,scrape_date`). The simulation passes the
  LLM a **rolling window of up to 5 periods**, **preferring quarterly** and only
  falling back to the annual series for cutoffs where screener has no quarterly
  history — never mixing the two.

- **News → sentiment — [`backtesting/fetch_news.py`](fetch_news.py)** (Wayback CDX
  → FinBERT). Writes the local news workbook
  (`Data/archive/backtest_news.xlsx`: sheets `News`, `Sentiment`, `Manifest`).
  Aggregates into 7-day sentiment windows the LLM consumes. Requires
  `BACKTEST_ENABLED=true`; skips `(target, year)` pairs already in the manifest.

Useful variants:

```bash
python -m backtesting.collect_data --only fundamentals     # skip news
python -m backtesting.collect_data --only news             # skip fundamentals
python -m backtesting.collect_data --tickers TCS,RELIANCE  # subset
```

**Fundamentals only (direct):**

```bash
python -m backtesting.fetch_fundamentals                   # full universe
python -m backtesting.fetch_fundamentals --dry-run         # parse + print, no write
python -m backtesting.fetch_fundamentals --no-annual       # quarterly only
python -m backtesting.fetch_fundamentals --tickers TCS --quarterly-lag-days 45 --annual-refresh-month 6
```

**News only (direct):**

```bash
python -m backtesting.fetch_news                           # full universe/window
python -m backtesting.fetch_news --tickers TCS --years 2022,2023
```

> **Coverage note.** screener.in gives ~3 years of *quarterly* data plus ~10 years
> of *annual* data, so the annual series covers the whole 2020-2026 window while
> quarterly detail is richer in later years. Any cutoff with no visible
> fundamentals (or no news in a window) degrades to `"unavailable"` in the LLM
> prompt — exactly what production does when data is missing.

---

## 2. Run the simulation

[`backtesting/walk_forward.py`](walk_forward.py) — the event-driven replay:
PIT-sliced frames → cold-start/warm-start retrain every 15 trading days →
T+1..T+15 ensemble forecast → GLM BUY/HOLD/AVOID → Markowitz/Sharpe sizing with a
cash reserve → orders filled at **next-bar open**; sentiment is re-checked every 7
days and only changed names are re-decided. Every event is written to the
bookkeeping workbook and a resumable state checkpoint.

Every **BUY** carries a **target price** (`sell_price`) and a **stoploss** — the
LLM layer now enforces both on a BUY (a BUY missing either is retried). The
trading algorithm rests those as exit orders: [`backtesting/exits.py`](exits.py)
checks each held name's daily O/H/L/C and books the exit the day its target or
stop is touched (conservative fills; a bar spanning both resolves to the stop),
holding the freed cash until a fresh suggestion. Exits land in the `Exits` /
`Price_Monitor` sheets.

**Validate on a short slice first** (recommended — confirms the whole chain incl.
live GLM before a long run):

```bash
python -m backtesting.walk_forward --start 2020-01-01 --end 2020-06-30
```

**Full run:**

```bash
python -m backtesting.walk_forward --start 2020-01-01 --end 2026-06-30
```

Flags: `--no-resume` (start fresh; default resumes from the last checkpoint),
`--device auto|cpu|cuda|mps`, `--limit-bars N` (smoke test).

Outputs:
- `outputs/backtest/state/backtest_state.json` — resumable book (cash, positions, equity curve, all logs).
- `outputs/backtest/backtest_bookkeeping.xlsx` — sheets: `LLM_Suggestions`,
  `Allocations`, `Reallocations`, `Price_Monitor`, `Exits`, `Trades`,
  `EquityCurve`, `Forecasts`, `PnL_Statement`, `Open_Positions`.
- `outputs/backtest/report/` — `metrics.json` + charts (equity vs NIFTY, drawdown,
  forecast MAE by horizon).
- `outputs/backtest/Saved_Models/` — the **backtest** model (production model is never touched).

---

## 3. Build the benchmark result sheet

[`backtesting/results.py`](results.py) — reads the run's equity curve and produces
one shareable workbook comparing the strategy to **NIFTY 50**, **Sensex** and
**government bonds**, all rebased to the ₹1-crore start.

```bash
python -m backtesting.results
python -m backtesting.results --bond-yield 0.072            # set the G-Sec proxy yield
python -m backtesting.results --bond-symbol <ETF>           # use a real bond ETF instead
```

- NIFTY 50 (`^NSEI`) and Sensex (`^BSESN`) are pulled via yfinance for the run
  window (NIFTY falls back to the local index cache).
- Government bonds = a buy-and-hold G-Sec proxy: a constant annual yield
  (default **7.0%**) compounded across the run's trading days. Override with a real
  ETF via `--bond-symbol`.

Output `outputs/backtest/results.xlsx`:
- `Summary` — ending value, total return, CAGR, annual vol, Sharpe, Sortino, max
  drawdown, Calmar for each track.
- `Benchmark_Stats` — strategy alpha/beta/up-down capture vs NIFTY and Sensex.
- `Equity` — daily aligned value of every track.

Plus `outputs/backtest/results_equity_vs_benchmarks.png`.

---

## Full sequence (copy-paste)

```bash
# setup env (see §0), then:
python -m features.cross_sectional --output Data/archive/indices.parquet --start 2019-01-01  # optional
python -m backtesting.collect_data                                   # news + fundamentals
python -m backtesting.walk_forward --start 2020-01-01 --end 2020-06-30   # validate slice
python -m backtesting.walk_forward --start 2020-01-01 --end 2026-06-30   # full run
python -m backtesting.results                                        # benchmark sheet
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GLM_API_KEY not set` / signals fail | Export `GLM_API_KEY` (the LLM decision layer needs it). |
| screener.in returns HTTP 403/429 | Set `SCREENER_COOKIE` from a logged-in browser session; increase `--pause`. |
| News step says "disabled" | Set `BACKTEST_ENABLED=true` (fundamentals run without it; news needs it). |
| No NIFTY line in the sim report | Build the index cache (`features.cross_sectional`); `results` fetches live regardless. |
| Fundamentals empty for early years | Expected where screener lacks history — those cutoffs read `"unavailable"` (PIT-correct). |
| Resume from a stopped run | Just re-run the same `walk_forward` command (it resumes from the checkpoint); add `--no-resume` to restart. |
