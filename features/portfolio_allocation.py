"""Production fund-allocation layer — advises users how much to invest per stock.

Production already produces the discrete LLM BUY/HOLD/AVOID call
(``features.trade_suggestions``). This module is the next step: it turns those
calls into concrete rupee sizing using the **same** quantitative engine the
backtest validates (:mod:`allocation.engine`), and writes the advice to a local
Excel workbook for the user.

Guarantees carried over from the engine:

* The book is only ever *partially* deployed (a cash reserve is always retained),
  so a recommendation can never draw the account to zero.
* **No new funds are ever invented and cash never goes negative.** When a BUY
  cannot be funded from free cash, capital is only rotated out of an existing
  holding if the BUY beats it on risk-reward by the configured edge — otherwise
  the BUY is downsized to the cash actually available.

Inputs can be injected (fully offline/testable) or loaded from the live
production surfaces (Firestore suggestions, the operational Sheet forecasts, the
archive workbook for the covariance history, and a local holdings state file).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from allocation.config import AllocationConfig
from allocation.engine import (
    OptimizeResult,
    ReconcileResult,
    optimize_weights,
    reconcile_orders,
    risk_reward_ratio,
)
from allocation.engine import BuyMeta, PositionView

DEFAULT_HOLDINGS_STATE = Path("state") / "portfolio_holdings.json"
DEFAULT_BOOKKEEPING = Path("outputs") / "production_allocation.xlsx"
FORECAST_HORIZON = "T+15"

# Final, website-facing suggestion collection (one doc per stock). BUY docs carry
# the target price, stop loss and % of funds to allocate + the LLM reason; AVOID
# docs carry the reason (and EXIT order if held). HOLD docs are stored with
# ``display=false`` so the front-end can ignore them while still having the data.
FINAL_SUGGESTIONS_COLLECTION = os.environ.get(
    "FINAL_SUGGESTIONS_FIRESTORE_COLLECTION", "final_suggestions"
)
FINAL_SUGGESTIONS_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------

@dataclass
class Holdings:
    capital: float                                   # total investable capital
    cash: float                                       # un-deployed cash
    positions: Dict[str, Dict[str, float]] = field(default_factory=dict)  # ticker -> {qty, avg_price}

    @classmethod
    def all_cash(cls, capital: float) -> "Holdings":
        return cls(capital=float(capital), cash=float(capital), positions={})

    @classmethod
    def load(cls, path: Path, *, capital: float) -> "Holdings":
        path = Path(path)
        if path.exists():
            data = json.loads(path.read_text())
            return cls(
                capital=float(data.get("capital", capital)),
                cash=float(data.get("cash", data.get("capital", capital))),
                positions=data.get("positions", {}),
            )
        return cls.all_cash(capital)

    def equity(self, prices: Dict[str, float]) -> float:
        total = float(self.cash)
        for t, pos in self.positions.items():
            px = float(prices.get(t.upper(), pos.get("avg_price", 0.0)) or 0.0)
            total += float(pos.get("qty", 0.0)) * px
        return total


@dataclass
class AllocationAdvice:
    ticker: str
    action: str
    target_weight: float          # fraction of equity
    target_amount: float          # ₹ to hold in this name
    current_amount: float         # ₹ currently held
    recommended_order: str        # "BUY ₹x" | "SELL ₹y" | "HOLD" | "EXIT"
    order_amount: float
    order_shares: int             # whole shares to trade (+buy / −sell); no fractions
    risk_reward: float
    confidence: Optional[float]
    expected_return_15d: Optional[float]
    rationale: Optional[str] = None
    funded_by: List[str] = field(default_factory=list)


@dataclass
class AllocationPlan:
    advice: List[AllocationAdvice]
    reallocations: List[Dict[str, Any]]
    projected_cash: float
    equity: float
    notes: List[str]
    opt: OptimizeResult
    recon: ReconcileResult
    suggestions: Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Core: compute the allocation (pure, engine-driven)
# ---------------------------------------------------------------------------

def compute_allocation(
    *,
    suggestions: Dict[str, Dict[str, Any]],
    forecasts: Dict[str, Dict[str, Any]],
    returns_hist: Dict[str, List[float]],
    holdings: Holdings,
    prices: Dict[str, float],
    config: AllocationConfig,
) -> AllocationPlan:
    """Size the current BUY set and reconcile against the user's holdings.

    ``suggestions[ticker]`` = the stored LLM doc (action/buy_price/sell_price/
    stoploss/confidence/rationale). ``forecasts[ticker]`` = ``{close, forecast:{
    T+..}}``. ``prices`` = current mark per ticker (defaults to the forecast close).
    """
    prices = {k.upper(): float(v) for k, v in prices.items() if v}
    for t, fc in forecasts.items():
        prices.setdefault(t.upper(), float(fc.get("close") or 0.0))

    buy_set, hold_set, avoid_set = _action_sets(suggestions)
    equity = holdings.equity(prices)

    # μ from the 15-day forecast path; confidence from the LLM doc.
    mu = {}
    for t in buy_set:
        fc = forecasts.get(t) or forecasts.get(t.upper()) or {}
        path = fc.get("forecast") or {}
        close = fc.get("close") or prices.get(t.upper())
        px = path.get(FORECAST_HORIZON)
        if px and close and close > 0:
            mu[t.upper()] = float(px) / float(close) - 1.0
    confidence = {t.upper(): float((suggestions.get(t) or {}).get("confidence") or 0.5) for t in buy_set}

    # Conviction-scaled deployment: size the BUY set with Markowitz, but deploy only
    # a floor for mediocre risk-reward (keep cash) and exhaust the book only for
    # excellent deals. The reconciler keeps the resulting reserve, freezes HOLDs,
    # and rotates capital out of weaker holdings when a BUY strictly beats them.
    names = [t.upper() for t in buy_set if t.upper() in mu and t.upper() in returns_hist]
    all_meta = _buy_meta(names, suggestions, forecasts, prices)
    risk_reward = {t: m.risk_reward for t, m in all_meta.items()}
    opt = optimize_weights(names, mu, confidence, returns_hist, config, risk_reward=risk_reward)

    positions_view = _position_views(holdings, prices, suggestions)
    buy_meta = _buy_meta(list(opt.weights), suggestions, forecasts, prices)
    target_notional = {t: w * equity for t, w in opt.weights.items() if w > 0}
    reserve = max(0.0, (1.0 - float(opt.deploy_fraction)) * equity)
    recon = reconcile_orders(
        cash=holdings.cash, positions=positions_view, target_notional=target_notional,
        buy_meta=buy_meta, avoid=avoid_set, hold=hold_set, config=config, equity=equity,
        min_cash_reserve=reserve,
    )

    advice = _build_advice(
        suggestions, opt, recon, positions_view, target_notional, prices, buy_meta, equity, hold_set,
    )
    return AllocationPlan(
        advice=advice, reallocations=recon.reallocations, projected_cash=recon.projected_cash,
        equity=equity, notes=recon.notes, opt=opt, recon=recon, suggestions=suggestions,
    )


def _action_sets(suggestions: Dict[str, Dict[str, Any]]):
    buy, hold, avoid = [], [], []
    for t, s in suggestions.items():
        action = str((s or {}).get("action", "")).upper()
        if action == "BUY":
            buy.append(t.upper())
        elif action == "HOLD":
            hold.append(t.upper())
        elif action == "AVOID":
            avoid.append(t.upper())
    return buy, hold, avoid


def _position_views(holdings: Holdings, prices: Dict[str, float], suggestions):
    views = {}
    for t, pos in holdings.positions.items():
        qty = float(pos.get("qty", 0.0))
        if qty <= 0:
            continue
        price = float(prices.get(t.upper(), pos.get("avg_price", 0.0)) or 0.0)
        s = suggestions.get(t.upper(), {}) or {}
        rr = risk_reward_ratio(s.get("buy_price") or pos.get("avg_price"), s.get("sell_price"), s.get("stoploss"))
        views[t.upper()] = PositionView(t.upper(), qty, price, rr, str(s.get("action", "HOLD")))
    return views


def _buy_meta(names, suggestions, forecasts, prices):
    meta = {}
    for t in names:
        s = suggestions.get(t.upper(), {}) or {}
        price = s.get("buy_price") or (forecasts.get(t.upper(), {}) or {}).get("close") or prices.get(t.upper())
        if not price or price <= 0:
            continue
        rr = risk_reward_ratio(price, s.get("sell_price"), s.get("stoploss"))
        meta[t.upper()] = BuyMeta(t.upper(), float(price), rr, float(s.get("confidence") or 0.5))
    return meta


def _build_advice(suggestions, opt, recon, positions_view, target_notional, prices, buy_meta, equity, hold_set):
    order_by_ticker: Dict[str, Any] = {}
    for o in recon.orders:
        order_by_ticker.setdefault(o.ticker, []).append(o)

    advice: List[AllocationAdvice] = []
    tickers = set(suggestions) | set(target_notional) | set(positions_view)
    for t in sorted(tickers):
        tu = t.upper()
        s = suggestions.get(t, {}) or suggestions.get(tu, {}) or {}
        action = str(s.get("action", "HOLD")).upper()
        cur_qty = positions_view[tu].qty if tu in positions_view else 0.0
        cur_price = prices.get(tu, positions_view[tu].price if tu in positions_view else 0.0)
        current_amount = cur_qty * cur_price
        target_amount = float(target_notional.get(tu, current_amount if tu in hold_set else 0.0))
        weight = float(opt.weights.get(tu, target_amount / equity if equity > 0 else 0.0))

        orders = order_by_ticker.get(tu, [])
        buys = sum(o.notional for o in orders if o.side == "BUY")
        sells = sum(o.notional for o in orders if o.side == "SELL")
        buy_shares = sum(o.qty for o in orders if o.side == "BUY")
        sell_shares = sum(o.qty for o in orders if o.side == "SELL")
        net_shares = int(round(buy_shares - sell_shares))   # whole shares, +buy / −sell
        funded = sorted({f for o in orders for f in o.funded_by})
        if buys > 0:
            rec, amt = f"BUY ₹{buys:,.0f}", buys
        elif sells > 0 and action == "AVOID":
            rec, amt = f"EXIT ₹{sells:,.0f}", -sells
        elif sells > 0:
            rec, amt = f"TRIM ₹{sells:,.0f}", -sells
        elif action == "HOLD" and cur_qty > 0:
            rec, amt = "HOLD", 0.0
        else:
            rec, amt = "NO ACTION", 0.0

        meta = buy_meta.get(tu)
        advice.append(AllocationAdvice(
            ticker=tu, action=action, target_weight=round(weight, 6),
            target_amount=round(target_amount, 2), current_amount=round(current_amount, 2),
            recommended_order=rec, order_amount=round(amt, 2), order_shares=net_shares,
            risk_reward=round(float(getattr(meta, "risk_reward", positions_view[tu].risk_reward if tu in positions_view else 1.0)), 3),
            confidence=s.get("confidence"),
            expected_return_15d=round(float(opt.expected_return.get(tu)), 6) if tu in opt.expected_return else None,
            rationale=s.get("rationale"), funded_by=funded,
        ))
    return advice


# ---------------------------------------------------------------------------
# Bookkeeping (production Excel)
# ---------------------------------------------------------------------------

def write_bookkeeping(plan: AllocationPlan, path: Path, *, as_of: Optional[str] = None) -> Path:
    """Write the user-facing allocation workbook (LLM calls + fund allocation)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    as_of = as_of or pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    suggestions_rows = [{
        "as_of": as_of, "ticker": t.upper(), "action": (s or {}).get("action"),
        "buy_price": (s or {}).get("buy_price"), "sell_day": (s or {}).get("sell_day"),
        "sell_price": (s or {}).get("sell_price"), "stoploss": (s or {}).get("stoploss"),
        "confidence": (s or {}).get("confidence"), "rationale": (s or {}).get("rationale"),
    } for t, s in sorted(plan.suggestions.items())]

    alloc_rows = [{
        "as_of": as_of, "ticker": a.ticker, "action": a.action,
        "target_weight_pct": round(a.target_weight * 100, 3), "target_amount": a.target_amount,
        "current_amount": a.current_amount, "recommended_order": a.recommended_order,
        "order_amount": a.order_amount, "order_shares": a.order_shares,
        "risk_reward": a.risk_reward, "confidence": a.confidence,
        "expected_return_15d": a.expected_return_15d, "funded_by": ",".join(a.funded_by),
    } for a in plan.advice]

    realloc_rows = [{"as_of": as_of, **r} for r in plan.reallocations]

    deploy_frac = float(getattr(plan.opt, "deploy_fraction", float("nan")))
    weighted_rr = float(getattr(plan.opt, "weighted_risk_reward", float("nan")))
    summary_rows = [
        ("As of", as_of),
        ("Total equity", round(plan.equity, 2)),
        ("Projected cash (reserve)", round(plan.projected_cash, 2)),
        ("Deployed", round(plan.equity - plan.projected_cash, 2)),
        ("Conviction deploy fraction", round(deploy_frac, 4) if deploy_frac == deploy_frac else None),
        ("Portfolio risk-reward", round(weighted_rr, 4) if weighted_rr == weighted_rr else None),
        ("BUY recommendations", sum(1 for a in plan.advice if a.recommended_order.startswith("BUY"))),
        ("Capital rotations", len(plan.reallocations)),
        ("Notes", "; ".join(plan.notes) if plan.notes else ""),
    ]

    tmp = path.with_suffix(".xlsx.tmp")
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        _sheet(writer, "LLM_Suggestions", suggestions_rows)
        _sheet(writer, "Fund_Allocation", alloc_rows)
        _sheet(writer, "Reallocations", realloc_rows)
        pd.DataFrame(summary_rows, columns=["metric", "value"]).to_excel(writer, sheet_name="Summary", index=False)
    os.replace(tmp, path)
    return path


def _sheet(writer, name: str, rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(rows) if rows else pd.DataFrame({"info": [f"no {name} rows"]})
    df.to_excel(writer, sheet_name=name[:31], index=False)


# ---------------------------------------------------------------------------
# Final website-facing suggestions (Firestore)
# ---------------------------------------------------------------------------

def build_final_suggestion_docs(plan: AllocationPlan, *, as_of: Optional[str] = None) -> List[Dict[str, Any]]:
    """One doc per stock combining the LLM verdict with the fund allocation.

    * **BUY** → ``target_price``, ``stop_loss``, ``allocation_pct`` (% of funds),
      ``allocation_amount`` (₹), ``buy_price``, ``confidence`` and the LLM
      ``reason``; ``display=True``.
    * **AVOID** (sell/ignore) → the LLM ``reason`` (and ``recommended_order`` =
      EXIT when a position is held); ``display=True``.
    * **HOLD** → stored with ``display=False`` so the front-end ignores it while
      the data stays queryable.
    """
    from datetime import datetime, timezone

    as_of = as_of or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    now_iso = datetime.now(timezone.utc).isoformat()
    advice_by_ticker = {a.ticker.upper(): a for a in plan.advice}

    docs: List[Dict[str, Any]] = []
    for ticker, s in plan.suggestions.items():
        s = s or {}
        tu = str(ticker).upper()
        action = str(s.get("action", "")).upper()
        a = advice_by_ticker.get(tu)
        doc: Dict[str, Any] = {
            "ticker": tu,
            "action": action,
            "as_of_date": as_of,
            "generated_at": now_iso,
            "reason": s.get("rationale"),
            "confidence": s.get("confidence"),
            # HOLD is not shown on the website; BUY/AVOID are.
            "display": action in ("BUY", "AVOID"),
            "schema_version": FINAL_SUGGESTIONS_SCHEMA_VERSION,
        }
        if action == "BUY":
            doc.update({
                "buy_price": s.get("buy_price"),
                "target_price": s.get("sell_price"),
                "stop_loss": s.get("stoploss"),
                "sell_day": s.get("sell_day"),
                "allocation_pct": round(float(a.target_weight) * 100.0, 3) if a else 0.0,
                "allocation_amount": round(float(a.target_amount), 2) if a else 0.0,
                "recommended_order": a.recommended_order if a else None,
                "expected_return_15d": a.expected_return_15d if a else None,
                "risk_reward": a.risk_reward if a else None,
            })
        elif action == "AVOID":
            doc.update({
                "recommended_order": a.recommended_order if a else "AVOID",
            })
        docs.append(doc)
    return docs


def write_final_suggestions_to_firestore(
    plan: AllocationPlan,
    *,
    client: Optional[Any] = None,
    collection: Optional[str] = None,
    as_of: Optional[str] = None,
) -> int:
    """Write the combined LLM-verdict + allocation docs (one per stock, id=ticker)."""
    collection = collection or FINAL_SUGGESTIONS_COLLECTION
    if client is None:
        from ingestion._firestore import init_firestore_client
        client = init_firestore_client()
    docs = build_final_suggestion_docs(plan, as_of=as_of)
    coll = client.collection(collection)
    batch = client.batch()
    pending = written = 0
    for doc in docs:
        batch.set(coll.document(doc["ticker"]), doc)
        pending += 1
        written += 1
        if pending >= 200:
            batch.commit()
            batch = client.batch()
            pending = 0
    if pending:
        batch.commit()
    return written


# ---------------------------------------------------------------------------
# Production loaders + CLI
# ---------------------------------------------------------------------------

def load_production_inputs(
    *,
    sheet_id: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    workbook: Optional[Path] = None,
    capital: float,
    holdings_state: Path = DEFAULT_HOLDINGS_STATE,
) -> Dict[str, Any]:
    """Load live suggestions (Firestore), forecasts (Sheet), Σ history (workbook)."""
    from features.trade_suggestions import load_forecasts, SUGGESTIONS_COLLECTION
    from ingestion._firestore import init_firestore_client

    forecasts_raw = load_forecasts(sheet_id or os.environ.get("SHEET_ID", ""), tickers)
    forecasts = {
        t.upper(): {"close": v.get("close"),
                    "forecast": {k: val for k, val in (v.get("forecast") or {}).items()}}
        for t, v in forecasts_raw.items()
    }

    fs = init_firestore_client()
    suggestions: Dict[str, Dict[str, Any]] = {}
    for doc in fs.collection(SUGGESTIONS_COLLECTION).stream():
        data = doc.to_dict() or {}
        suggestions[str(data.get("ticker") or doc.id).upper()] = data

    returns_hist = _returns_from_workbook(workbook, list(forecasts))
    holdings = Holdings.load(holdings_state, capital=capital)
    prices = {t: (v.get("close") or 0.0) for t, v in forecasts.items()}
    return {"suggestions": suggestions, "forecasts": forecasts,
            "returns_hist": returns_hist, "holdings": holdings, "prices": prices}


def _returns_from_workbook(workbook: Optional[Path], tickers: List[str], lookback: int = 252) -> Dict[str, List[float]]:
    import numpy as np
    if workbook is None:
        workbook = Path("Data") / "archive" / "nse_stock_data_train_filled.xlsx"
    if not Path(workbook).exists():
        return {}
    from scripts.train_from_workbook import build_frames
    frames = build_frames(Path(workbook), tickers or None)
    out: Dict[str, List[float]] = {}
    for sym, frame in frames.items():
        close = pd.to_numeric(frame.get("Close"), errors="coerce").dropna().to_numpy(dtype=float)
        if len(close) < 3:
            continue
        rets = np.diff(np.log(np.clip(close, 1e-9, None)))
        rets = rets[np.isfinite(rets)]
        if len(rets) >= 2:
            out[sym.upper()] = rets[-lookback:].tolist()
    return out


def run(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Production fund-allocation advice (Markowitz + cash-guard).")
    p.add_argument("--capital", type=float, default=float(os.environ.get("ALLOC_INITIAL_CAPITAL", "1000000")))
    p.add_argument("--sheet-id", default=os.environ.get("SHEET_ID"))
    p.add_argument("--tickers", default=None)
    p.add_argument("--workbook", default=None)
    p.add_argument("--holdings-state", default=str(DEFAULT_HOLDINGS_STATE))
    p.add_argument("--output", default=str(DEFAULT_BOOKKEEPING))
    p.add_argument("--final-collection", default=FINAL_SUGGESTIONS_COLLECTION,
                   help="Firestore collection for the website-facing final suggestions.")
    p.add_argument("--no-firestore", action="store_true",
                   help="Skip writing the final suggestions to Firestore (Excel only).")
    args = p.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    inputs = load_production_inputs(
        sheet_id=args.sheet_id, tickers=tickers,
        workbook=Path(args.workbook) if args.workbook else None,
        capital=args.capital, holdings_state=Path(args.holdings_state),
    )
    config = AllocationConfig.from_env(initial_capital=args.capital)
    plan = compute_allocation(config=config, **inputs)
    out = write_bookkeeping(plan, Path(args.output))

    final_written = 0
    if not args.no_firestore:
        try:
            final_written = write_final_suggestions_to_firestore(plan, collection=args.final_collection)
        except Exception as exc:  # noqa: BLE001 - Firestore failure shouldn't lose the Excel output
            print(f"[allocation] final-suggestions Firestore write failed: {exc}", file=sys.stderr)

    n_buy = sum(1 for a in plan.advice if a.recommended_order.startswith("BUY"))
    n_avoid = sum(1 for a in plan.advice if str(a.action).upper() == "AVOID")
    summary = {
        "equity": round(plan.equity, 2), "projected_cash": round(plan.projected_cash, 2),
        "buy_recommendations": n_buy, "avoid_recommendations": n_avoid,
        "reallocations": len(plan.reallocations), "bookkeeping": str(out),
        "final_suggestions_written": final_written, "final_collection": args.final_collection,
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
    sys.exit(0)
