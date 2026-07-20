"""Walk-forward backtest entrypoint — the event-driven simulation loop.

Replays the exact production pipeline (PIT-sliced frames → ≤C ensemble forecast →
GLM BUY/HOLD/AVOID → quantitative Markowitz sizing → cash-guarded orders filled at
next-bar open) day by day over the configured universe, recording every artefact
to the local bookkeeping workbook and a resumable state checkpoint.

CLI::

    python -m backtesting.walk_forward --start 2020-01-01 --end 2020-06-30
    python -m backtesting.walk_forward --start 2020-01-01 --end 2026-06-30 --resume
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd

from allocation.config import AllocationConfig
from app.config.backtest_settings import BacktestSettings, get_backtest_settings
from backtesting import calendar as bt_calendar
from backtesting import costs as bt_costs
from backtesting import data_access
from backtesting import exits as bt_exits
from backtesting import model_cycle
from backtesting import portfolio as bt_portfolio
from backtesting import report as bt_report
from backtesting import sentiment_store
from backtesting import signals as bt_signals
from backtesting.bookkeeping import BacktestBookkeeper
from backtesting.state import BacktestState


def log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)


def _action_sets(last_signal: Dict[str, Dict[str, Any]]):
    buy, hold, avoid = [], [], []
    for t, s in last_signal.items():
        action = str((s or {}).get("action", "")).upper()
        if action == "BUY":
            buy.append(t.upper())
        elif action == "HOLD":
            hold.append(t.upper())
        elif action == "AVOID":
            avoid.append(t.upper())
    return buy, hold, avoid


def _atr_map(frames: Dict[str, pd.DataFrame], cutoff: Any) -> Dict[str, float]:
    truncated = data_access.truncate_frames(frames, cutoff)
    return {sym: bt_costs.atr14_from_frame(frame) for sym, frame in truncated.items()}


def _orders_to_dicts(orders) -> List[Dict[str, Any]]:
    return [
        {"ticker": o.ticker, "side": o.side, "qty": o.qty, "price": o.price,
         "notional": o.notional, "reason": o.reason, "funded_by": list(o.funded_by)}
        for o in orders
    ]


def _order_views(dicts: List[Dict[str, Any]]):
    return [SimpleNamespace(**d) for d in dicts]


def _benchmark_close_at(bench: pd.DataFrame, cutoff: Any, symbol: str) -> Optional[float]:
    if bench is None or bench.empty:
        return None
    C = pd.Timestamp(cutoff).normalize()
    sub = bench.loc[bench.index <= C]
    if sub.empty:
        return None
    col = symbol if symbol in bench.columns else ("Close" if "Close" in bench.columns else bench.columns[0])
    try:
        return float(sub.iloc[-1][col])
    except Exception:
        return None


def run_backtest(
    settings: Optional[BacktestSettings] = None,
    *,
    resume: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
    device: str = "cpu",
    limit_bars: Optional[int] = None,
) -> Dict[str, Any]:
    settings = settings or get_backtest_settings()
    config = AllocationConfig.from_backtest_settings(settings)
    from features.entry_gates import GateConfig
    gate_cfg = (GateConfig.from_backtest_settings(settings)
                if getattr(settings, "entry_gates_enabled", True) else None)
    start = start or settings.start_date
    end = end or (settings.end_date or None)

    from scripts.train_from_workbook import build_frames
    tickers = settings.resolved_tickers()
    log(f"[backtest] building frames from {settings.workbook_path} for {len(tickers)} tickers …")
    frames = build_frames(Path(settings.workbook_path), tickers or None)
    universe = sorted(frames.keys())
    log(f"[backtest] {len(universe)} symbol frames built")

    days = bt_calendar.trading_days(frames, start, end)
    # The second cadence is the WEEKLY open-position review (5 trading bars ≈ 1
    # calendar week): between retrains, only held names are re-underwritten.
    sched = bt_calendar.cadence_schedule(days, settings.retrain_interval_days,
                                         getattr(settings, "review_interval_days", 5))
    log(f"[backtest] {len(days)} trading bars in [{start}, {end}]")

    try:
        from features.cross_sectional import load_index_history
        bench = load_index_history(Path(settings.index_cache_path))
    except Exception:  # pragma: no cover
        bench = pd.DataFrame()

    state = BacktestState.load_or_init(Path(settings.state_path), initial_capital=settings.initial_capital, resume=resume)
    bookkeeper = BacktestBookkeeper(Path(settings.bookkeeping_path))
    resume_from = pd.Timestamp(state.last_bar) if (resume and state.last_bar) else None

    news_wb = Path(settings.news_workbook_path) if Path(settings.news_workbook_path).exists() else None
    sent_cache = Path(settings.sentiment_cache_path) if Path(settings.sentiment_cache_path).exists() else None

    # Rotating LLM key pool (Ollama keys → Groq keys → wrap). Falls back to the
    # single production GLM_API_KEY/GLM_MODEL when no pool is configured.
    from backtesting.llm_client import from_settings as _build_llm_client
    llm = _build_llm_client(settings, logger=log)
    if llm is not None:
        log(f"[backtest] LLM key pool: {len(llm.providers)} providers "
            f"[{', '.join(p.name for p in llm.providers)}]")
    llm_ask = llm if llm is not None else bt_signals.ask_glm

    processed = 0
    for (C, events) in sched:
        if resume_from is not None and C <= resume_from:
            continue
        if limit_bars is not None and processed >= limit_bars:
            break
        processed += 1

        # 1) mark to close ≤ C, then settle last bar's orders at this bar's open.
        state.mark_to_market(data_access.close_asof(frames, C))
        opens = data_access.open_on(frames, C)
        fills = bt_portfolio.fill_at_open(
            _order_views(state.pending_orders_prev), opens,
            atr_map=_atr_map(frames, C),
            cost_bps=settings.cost_bps_roundtrip, slippage_atr_mult=settings.slippage_atr_mult,
        )
        state.apply_fills(fills, bar=C)
        state.pending_orders_prev = []

        # 1b) intraday exit management (hold-and-trail): a stop touch still sells
        # same-bar, but a *target* touch now arms/ratchets a trailing stop instead
        # of banking the profit (finding D: 62% of target exits kept rising), and
        # positions older than ``max_holding_days`` age out (finding E: the
        # 16–30d bucket was the only profitable one). Strictly PIT — only bar
        # C's own O/H/L/C is consulted (see backtesting.exits).
        ohlc = data_access.day_ohlc(frames, C)
        atr_now = _atr_map(frames, C)
        exits, stop_updates = bt_exits.scan_exits_and_trail(
            state.positions, state.last_signal, ohlc,
            honor_stop=settings.honor_stoploss,
            trail_on_target=settings.trail_on_target,
            trail_atr_mult=settings.trail_atr_mult,
            atr_map=atr_now,
            max_holding_days=settings.max_holding_days,
            bar=C,
        )
        for t, upd in stop_updates.items():
            sig = state.last_signal.get(t)
            if sig is not None:
                sig.update(upd)
        state.record_price_monitor(
            bt_exits.monitor_rows(C, state.positions, state.last_signal, ohlc, exits,
                                  honor_stop=settings.honor_stoploss,
                                  stop_updates=stop_updates)
        )
        if exits:
            exit_fills = bt_portfolio.fill_exits(
                exits, state.positions, atr_map=atr_now,
                cost_bps=settings.cost_bps_roundtrip, slippage_atr_mult=settings.slippage_atr_mult,
            )
            fill_by_ticker = {f.ticker.upper(): f for f in exit_fills}
            state.apply_fills(exit_fills, bar=C)
            for ex in exits:
                f = fill_by_ticker.get(ex.ticker.upper())
                if f is not None:
                    state.record_exit(C, ex, f)
                state.mark_exited(ex.ticker)      # hold cash until the next suggestion

        targets: Optional[List[str]] = None
        frames_asof = data_access.truncate_frames(frames, C)

        if events.get("retrain"):
            prev_cutoff = state.models_cutoff
            if prev_cutoff is None:
                log(f"[backtest] {C.date()} cold-start seed …")
                model_cycle.seed_train_at(frames, C, settings, device=device)
            else:
                log(f"[backtest] {C.date()} warm-start retrain (since {prev_cutoff}) …")
                model_cycle.retrain_at(frames, C, settings, prev_cutoff=prev_cutoff, device=device)
            fc = model_cycle.forecast_at(frames, C, settings, device=device)
            state.last_forecast = fc
            state.models_cutoff = C.strftime("%Y-%m-%d")
            _log_forecasts(state, fc, frames_asof, C)

            sent = sentiment_store.window_asof(universe, C, settings.sentiment_interval_days,
                                               workbook_path=news_wb, cache_path=sent_cache)
            funds = bt_signals.fundamentals_asof(universe, C, reporting_lag_days=settings.reporting_lag_days)
            new = bt_signals.run_signals(universe, fc, frames_asof, sent, funds, C, ask=llm_ask,
                                         positions=_review_payloads(state, C, atr_now, settings),
                                         gate_config=gate_cfg)
            _merge_signals(state, new, sent, universe)
            bookkeeper.log_suggestions(C, new)
            targets = list(new.keys())

        elif events.get("sentiment"):
            # Weekly open-position review: ONLY held names are re-underwritten
            # (fresh sentiment + the latest forecasts through the position-review
            # prompt) so a broken thesis is caught within ~a week instead of
            # waiting for the next retrain. Fresh entries are decided on retrain
            # bars alone — no new BUY signals are generated here.
            held_now = sorted({t.upper() for t, p in state.positions.items()
                               if float(p.get("qty", 0)) > 0})
            if held_now:
                sent = sentiment_store.window_asof(held_now, C, settings.sentiment_interval_days,
                                                   workbook_path=news_wb, cache_path=sent_cache)
                funds = bt_signals.fundamentals_asof(held_now, C, reporting_lag_days=settings.reporting_lag_days)
                new = bt_signals.run_signals(held_now, state.last_forecast, frames_asof, sent, funds, C,
                                             ask=llm_ask, positions=_review_payloads(state, C, atr_now, settings),
                                             gate_config=gate_cfg)
                _merge_signals(state, new, sent, held_now)
                bookkeeper.log_suggestions(C, new)
                targets = list(new.keys())

        # 2) rebalance: size the BUY set, reconcile to cash-guarded orders.
        if targets:
            # Time-diversification: at most N NEW names may enter per cycle.
            # The cap runs on the MERGED signal view so stale BUYs carried from
            # earlier cycles count too; overflow is demoted to HOLD in place and
            # re-qualifies when its next (fresh-forecast) BUY signal arrives.
            from features.entry_gates import apply_entry_cap
            held_names = {t.upper() for t, p in state.positions.items()
                          if float(p.get("qty", 0)) > 0}
            deferred = apply_entry_cap(
                state.last_signal, held_names,
                int(getattr(settings, "max_new_positions_per_cycle", 0)),
                cost_pct=settings.gate_roundtrip_cost_pct,
            )
            if deferred:
                log(f"[backtest] {C.date()} entry cap "
                    f"({settings.max_new_positions_per_cycle}): deferred {', '.join(deferred)}")
            buy_set, hold_set, avoid_set = _action_sets(state.last_signal)
            closes = data_access.close_asof(frames, C)
            equity = state.equity()
            returns_hist = data_access.returns_hist_asof(frames, C)
            confidence = {t: float((state.last_signal.get(t) or {}).get("confidence") or 0.5) for t in buy_set}
            # Per-name risk-reward drives conviction-scaled deployment: mediocre BUY
            # sets deploy only a floor (keep cash); excellent sets can exhaust it.
            buy_meta = bt_portfolio.build_buy_meta(buy_set, closes, state.last_signal)
            risk_reward = {t: m.risk_reward for t, m in buy_meta.items()}

            opt = bt_portfolio.optimize(
                buy_set, state.last_forecast, closes, returns_hist, confidence, config,
                risk_reward=risk_reward,
            )
            recon = bt_portfolio.reconcile(
                cash=state.cash, positions=state.positions, mark_prices=state.mark_prices,
                last_signal=state.last_signal, weights=opt.weights, equity=equity,
                avoid_set=avoid_set, hold_set=hold_set, config=config,
                deploy_fraction=opt.deploy_fraction,
            )
            state.pending_orders = _orders_to_dicts(recon.orders)
            bookkeeper.log_allocation(C, opt, equity, buy_meta)
            bookkeeper.log_reallocations(C, recon)

        # 3) queue this bar's orders for next-bar-open fill, mark equity, checkpoint.
        state.pending_orders_prev = state.pending_orders
        state.pending_orders = []
        state.record_equity(C, benchmark=_benchmark_close_at(bench, C, settings.benchmark_symbol))
        state.checkpoint(Path(settings.state_path))
        if processed % 10 == 0:
            bookkeeper.flush(state)

    _join_realized(state, frames, days)
    bench_series = _benchmark_series(bench, settings.benchmark_symbol)
    metrics = bt_report.write_report(state, Path(settings.report_dir), benchmark=bench_series)
    bookkeeper.flush(state)
    log(f"[backtest] done. equity={state.equity():,.0f} realized_pnl={state.realized_pnl:,.0f} "
        f"bars={processed} → {settings.bookkeeping_path}")
    return {"bars_processed": processed, "ending_equity": state.equity(),
            "metrics": metrics, "bookkeeping": str(settings.bookkeeping_path)}


def _review_payloads(state: BacktestState, C, atr_map=None, settings=None) -> Dict[str, Dict[str, Any]]:
    """Open-position blocks for the LLM position-review prompt.

    ``min_price_to_trail_inr`` is the profit cushion the trade must clear before
    its stop may be tightened to break-even — ``entry + max(atr_mult*ATR,
    profit_pct*entry)`` — so a young position keeps its wide entry stop and is
    not whipsawed out at break-even by normal daily noise.
    """
    atr_map = atr_map or {}
    profit_pct = float(getattr(settings, "trail_arm_profit_pct", 0.02)) if settings else 0.02
    atr_mult = float(getattr(settings, "trail_arm_atr_mult", 1.0)) if settings else 1.0
    out: Dict[str, Dict[str, Any]] = {}
    for t, pos in state.positions.items():
        qty = float(pos.get("qty", 0.0))
        if qty <= 0:
            continue
        tkr = t.upper()
        sig = state.last_signal.get(tkr) or {}
        entry_bar = pos.get("entry_bar")
        try:
            days_held = int((pd.Timestamp(C) - pd.Timestamp(entry_bar)).days) if entry_bar else None
        except Exception:
            days_held = None
        entry = float(pos.get("avg_price", 0.0))
        atr = float(atr_map.get(tkr) or atr_map.get(t) or 0.0)
        arm_level = entry + max(atr_mult * atr, profit_pct * entry) if entry > 0 else None
        out[tkr] = {
            "entry_price_inr": entry,
            "entry_date": entry_bar,
            "days_held": days_held,
            "qty": qty,
            "current_target": sig.get("sell_price"),
            "current_stoploss": sig.get("stoploss"),
            "trail_armed": bool(sig.get("trail_armed")),
            "current_price_inr": state.mark_prices.get(tkr),
            "min_price_to_trail_inr": round(arm_level, 4) if arm_level else None,
            "atr_inr": round(atr, 4) if atr > 0 else None,
        }
    return out


def _merge_signals(state: BacktestState, new: Dict[str, Dict[str, Any]], sent, tickers) -> None:
    for t, sig in new.items():
        if "error" in sig:
            continue
        prev = state.last_signal.get(t.upper()) or {}
        # A review HOLD keeps the trail armed and never loosens the stop.
        if prev.get("trail_armed") and str(sig.get("action", "")).upper() == "HOLD":
            sig["trail_armed"] = True
            prev_stop, new_stop = prev.get("stoploss"), sig.get("stoploss")
            try:
                if prev_stop is not None and (new_stop is None or float(new_stop) < float(prev_stop)):
                    sig["stoploss"] = prev_stop
            except (TypeError, ValueError):
                sig["stoploss"] = prev_stop
        state.last_signal[t.upper()] = sig
        state.signal_log.append({
            "bar": sig.get("as_of_date"), "ticker": t.upper(), "action": sig.get("action"),
            "confidence": sig.get("confidence"), "anchor_close": sig.get("last_close"),
            "buy_price": sig.get("buy_price"), "sell_price": sig.get("sell_price"),
            "stoploss": sig.get("stoploss"),
        })
    for t in tickers:
        state.last_sentiment_fp[t] = sentiment_store.sent_fingerprint(sent, t)


def _log_forecasts(state: BacktestState, fc: Dict[str, Dict[str, float]], frames_asof, C) -> None:
    for ticker, path in fc.items():
        frame = frames_asof.get(ticker)
        anchor = None
        if frame is not None and not frame.empty:
            closes = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            anchor = float(closes.iloc[-1]) if len(closes) else None
        for horizon, px in path.items():
            state.forecast_log.append({
                "bar": C.strftime("%Y-%m-%d"), "ticker": ticker.upper(),
                "horizon": horizon, "forecast": float(px), "anchor_close": anchor,
                "realized": None,
            })


def _join_realized(state: BacktestState, frames: Dict[str, pd.DataFrame], days: List[pd.Timestamp]) -> None:
    """Post-hoc: fill realized close h trading-days after each forecast (eval only)."""
    day_index = {d: i for i, d in enumerate(days)}

    def realized_close(sym: str, bar: str, h: int) -> Optional[float]:
        frame = frames.get(sym)
        if frame is None:
            return None
        try:
            bar_ts = pd.Timestamp(bar)
        except Exception:
            return None
        i = day_index.get(bar_ts)
        if i is None or i + h >= len(days):
            return None
        target_day = days[i + h]
        dser = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        row = frame.loc[dser == target_day]
        if row.empty:
            return None
        val = pd.to_numeric(row.iloc[-1].get("Close"), errors="coerce")
        return float(val) if pd.notna(val) else None

    for row in state.forecast_log:
        if row.get("realized") is not None:
            continue
        h = int(str(row["horizon"]).replace("T+", ""))
        row["realized"] = realized_close(row["ticker"], row["bar"], h)

    # realized_return for confidence calibration = 15-day forward close return.
    for s in state.signal_log:
        if s.get("realized_return") is not None or s.get("anchor_close") in (None, 0):
            continue
        rc = realized_close(s["ticker"], s["bar"], 15)
        if rc is not None and s["anchor_close"]:
            s["realized_return"] = rc / float(s["anchor_close"]) - 1.0


def _benchmark_series(bench: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
    if bench is None or bench.empty:
        return None
    col = symbol if symbol in bench.columns else ("Close" if "Close" in bench.columns else bench.columns[0])
    try:
        return bench[col].dropna()
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward backtest harness (Phase 4).")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--limit-bars", type=int, default=None, help="Cap bars processed (smoke test).")
    args = p.parse_args(argv)
    result = run_backtest(resume=args.resume, start=args.start, end=args.end,
                          device=args.device, limit_bars=args.limit_bars)
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
