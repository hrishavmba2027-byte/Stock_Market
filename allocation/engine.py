"""Pure fund-allocation math: Markowitz sizing + cash-guard reconciliation.

Nothing here touches Google Sheets, Firestore, torch, the network, or any config
singleton — every function takes plain numbers/arrays/dataclasses so it is fully
unit-testable and reused identically by the backtest and by production.

Two stages:

* :func:`optimize_weights` — size the BUY set. Expected returns come from the
  15-day forecast path (``forecast["T+15"]/close - 1``), tilted by LLM
  confidence; covariance is Ledoit-Wolf shrunk; per-name caps are vol-scaled; a
  max-Sharpe weight vector is solved under a cash reserve so the deployable
  budget is ``1 - cash_floor_frac`` of equity (partial deployment by design).
* :func:`reconcile_orders` — turn target notionals into BUY/SELL orders while
  guaranteeing the book never goes negative and **no new funds are ever
  invented**. When free cash cannot fund a BUY, capital is rotated out of an
  existing holding *only if* the BUY beats that holding on risk-reward by
  ``reallocation_edge``; otherwise the BUY is downsized to the cash actually
  available (or dropped below the min ticket).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _whole_shares(qty: float, *, side: str) -> int:
    """Round a share quantity to a whole number — no fractional shares.

    BUYs floor (you can only buy as many whole shares as the budget affords;
    rounding up would overspend the funded amount and could overdraw cash).
    SELLs round up (raise at least the shares needed), and every caller caps the
    result at the shares actually held, so a round-up can never oversell.
    """
    if qty <= 0:
        return 0
    return int(math.floor(qty)) if side == "BUY" else int(math.ceil(qty - 1e-9))

# scipy / scikit-learn are declared deps (see the plan), but the engine degrades
# gracefully to closed-form / sample-covariance fallbacks if either is absent so
# the pure math never hard-fails an import.
try:  # pragma: no cover - exercised indirectly
    from scipy.optimize import minimize as _scipy_minimize
except Exception:  # pragma: no cover
    _scipy_minimize = None

try:  # pragma: no cover
    from sklearn.covariance import LedoitWolf as _LedoitWolf
except Exception:  # pragma: no cover
    _LedoitWolf = None


# ---------------------------------------------------------------------------
# Expected returns, covariance, caps
# ---------------------------------------------------------------------------

def tilt_expected_returns(
    mu: Sequence[float],
    confidence: Sequence[float],
    tilt: float,
) -> np.ndarray:
    """``mu_i * (1 + tilt*(2*conf_i - 1))`` — a mild confidence tilt.

    A confidence of 0.5 leaves ``mu`` unchanged; 1.0 scales it up by ``tilt`` and
    0.0 scales it down by ``tilt``. The multiplier is clipped so it can never be
    negative — the tilt adjusts magnitude, it never flips the sign of a forecast.
    """
    mu_arr = np.asarray(mu, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    if conf.shape != mu_arr.shape:
        conf = np.full_like(mu_arr, 0.5)
    conf = np.clip(conf, 0.0, 1.0)
    multiplier = np.clip(1.0 + float(tilt) * (2.0 * conf - 1.0), 0.0, None)
    return mu_arr * multiplier


def shrunk_covariance(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance of a ``(n_obs, n_assets)`` return matrix.

    Falls back to a ridge-regularised sample covariance when scikit-learn is
    unavailable or the sample is too short to shrink, so the result is always a
    finite, positive-definite matrix that ``max_sharpe_weights`` can invert.
    """
    X = np.asarray(returns, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("returns must be a 2-D (n_obs, n_assets) matrix")
    n_assets = X.shape[1]
    if X.shape[0] < 2:
        return np.eye(n_assets)
    if _LedoitWolf is not None and X.shape[0] >= 3:
        try:
            cov = _LedoitWolf(assume_centered=False).fit(X).covariance_
            return _make_pd(np.asarray(cov, dtype=np.float64))
        except Exception:  # pragma: no cover - numerical edge cases
            pass
    sample = np.cov(X, rowvar=False)
    sample = np.atleast_2d(sample)
    return _make_pd(sample)


def _make_pd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Nudge a covariance matrix to be symmetric positive-definite."""
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov)
    floor = max(eps, float(np.nanmean(np.abs(diag))) * 1e-6) if diag.size else eps
    cov = cov + np.eye(cov.shape[0]) * floor
    return cov


def per_name_caps(
    sigma_annual: Sequence[float],
    per_name_cap_frac: float,
    vol_target_annual: float,
) -> np.ndarray:
    """Vol-scaled per-name weight ceilings.

    ``cap_i = min(per_name_cap_frac, vol_target_annual / sigma_i)`` — a volatile
    name is capped tighter so no single position dominates portfolio risk. A
    zero/degenerate sigma falls back to the flat ``per_name_cap_frac``.
    """
    sigma = np.asarray(sigma_annual, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_cap = np.where(sigma > 1e-9, float(vol_target_annual) / sigma, float(per_name_cap_frac))
    caps = np.minimum(float(per_name_cap_frac), vol_cap)
    return np.clip(caps, 0.0, float(per_name_cap_frac))


def _sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    var = float(w @ cov @ w)
    if var <= 0:
        return 0.0
    return float((w @ mu - rf) / np.sqrt(var))


def max_sharpe_weights(
    mu: Sequence[float],
    cov: np.ndarray,
    caps: Sequence[float],
    budget: float,
    *,
    rf: float = 0.0,
) -> np.ndarray:
    """Long-only max-Sharpe weights summing to ``budget`` under ``0 <= w_i <= cap_i``.

    Solved with SLSQP; if scipy is missing or the solve fails/degenerates it falls
    back to the closed-form tangency portfolio ``w ∝ Σ⁻¹μ`` (clipped to the caps
    and renormalised to the budget). When every ``mu`` is non-positive there is no
    attractive long portfolio, so all-cash (zero weights) is returned.
    """
    mu_arr = np.asarray(mu, dtype=np.float64)
    caps_arr = np.asarray(caps, dtype=np.float64)
    n = mu_arr.shape[0]
    budget = float(budget)
    if n == 0 or budget <= 0:
        return np.zeros(n)
    if float(caps_arr.sum()) < budget - 1e-12:
        # Caps cannot absorb the full budget; scale the target down to what the
        # caps allow (the remainder simply stays in cash).
        budget = float(caps_arr.sum())
    if budget <= 0 or not np.any(mu_arr > 0):
        return np.zeros(n)

    closed = _closed_form_tangency(mu_arr, cov, caps_arr, budget, rf)
    if _scipy_minimize is None:
        return closed

    x0 = closed if closed.sum() > 0 else _feasible_start(caps_arr, budget)
    bounds = [(0.0, float(c)) for c in caps_arr]
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - budget)},)

    def neg_sharpe(w: np.ndarray) -> float:
        return -_sharpe(w, mu_arr, cov, rf)

    try:
        res = _scipy_minimize(
            neg_sharpe, x0, method="SLSQP", bounds=bounds,
            constraints=constraints, options={"maxiter": 300, "ftol": 1e-9},
        )
        w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, caps_arr)
        if res.success and w.sum() > 0 and np.isfinite(w).all():
            w = _renormalise(w, caps_arr, budget)
            # Keep whichever candidate actually has the higher Sharpe.
            if _sharpe(w, mu_arr, cov, rf) >= _sharpe(closed, mu_arr, cov, rf):
                return w
    except Exception:  # pragma: no cover - solver hiccups fall back
        pass
    return closed


def _closed_form_tangency(
    mu: np.ndarray, cov: np.ndarray, caps: np.ndarray, budget: float, rf: float
) -> np.ndarray:
    excess = mu - rf
    try:
        raw = np.linalg.solve(cov, excess)
    except np.linalg.LinAlgError:  # pragma: no cover
        raw = np.linalg.pinv(cov) @ excess
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 0:
        return np.zeros_like(mu)
    w = raw / raw.sum() * budget
    return _renormalise(np.clip(w, 0.0, caps), caps, budget)


def _feasible_start(caps: np.ndarray, budget: float) -> np.ndarray:
    if caps.sum() <= 0:
        return np.zeros_like(caps)
    return caps / caps.sum() * budget


def _renormalise(w: np.ndarray, caps: np.ndarray, budget: float, iters: int = 64) -> np.ndarray:
    """Adjust ``w`` to sum to ``budget`` while respecting ``0 <= w_i <= cap_i``.

    Water-filling: the residual ``budget - sum(w)`` is repeatedly distributed
    across the names that still have room (headroom up to the cap when adding,
    or weight down to zero when removing), in proportion to that room. This
    converges to an exact fill unless the caps physically cannot absorb the
    budget, in which case ``w`` saturates at the caps.
    """
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, caps)
    for _ in range(iters):
        total = float(w.sum())
        if total <= 0:
            return w
        residual = budget - total
        if abs(residual) < 1e-12:
            break
        room = (caps - w) if residual > 0 else w.copy()
        room = np.clip(room, 0.0, None)
        room_sum = float(room.sum())
        if room_sum <= 1e-15:
            break
        w = np.clip(w + residual * room / room_sum, 0.0, caps)
    return w


def risk_reward_ratio(
    entry: Optional[float],
    target: Optional[float],
    stop: Optional[float],
    *,
    default: float = 1.0,
) -> float:
    """Reward-to-risk of a suggestion = upside%/downside% from entry.

    ``reward = (target - entry)/entry``; ``risk = (entry - stop)/entry``. When the
    inputs are missing or the risk leg is non-positive (no stop below entry), a
    conservative ``default`` is returned so ranking still works. The ratio is the
    metric used to decide whether a new BUY earns the right to pull capital out of
    an existing holding.
    """
    try:
        e = float(entry)
        t = float(target)
        s = float(stop)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite([e, t, s]).all() or e <= 0:
        return float(default)
    reward = (t - e) / e
    risk = (e - s) / e
    if risk <= 1e-9:
        return float(default)
    if reward <= 0:
        return 0.0
    return float(reward / risk)


# ---------------------------------------------------------------------------
# Weight optimisation entry point (assembles μ/Σ/caps and solves)
# ---------------------------------------------------------------------------

def deployment_fraction(weighted_rr: float, config) -> float:
    """Conviction-scaled fraction of the book to deploy, driven by risk-reward.

    A linear ramp: at/below ``rr_min_deploy`` (reward ≈ risk — a marginal deal) we
    deploy only ``deploy_floor_frac`` and keep the rest in cash; at/above
    ``rr_full_deploy`` (an excellent deal) we deploy the whole book. This is the
    "partial allocation, keep money at hand, and only exhaust the cash when the
    deal looks really good" rule. Returns a fraction in ``[deploy_floor_frac, 1]``.
    """
    floor = float(getattr(config, "deploy_floor_frac", 0.30))
    lo = float(getattr(config, "rr_min_deploy", 1.0))
    hi = float(getattr(config, "rr_full_deploy", 3.0))
    if not np.isfinite(weighted_rr):
        return floor
    if hi <= lo:
        return 1.0 if weighted_rr >= hi else floor
    frac = floor + (float(weighted_rr) - lo) / (hi - lo) * (1.0 - floor)
    return float(min(1.0, max(floor, frac)))


@dataclass
class OptimizeResult:
    weights: Dict[str, float]           # ticker -> fraction of equity
    expected_return: Dict[str, float]   # tilted μ per BUY name
    sigma_annual: Dict[str, float]      # annualised vol per BUY name
    cap: Dict[str, float]               # per-name cap used
    deployable_budget: float            # fraction of equity available to deploy
    deploy_fraction: float = 1.0        # conviction-scaled fraction actually deployed
    weighted_risk_reward: float = float("nan")  # portfolio RR that set the deployment


def optimize_weights(
    buy_set: Sequence[str],
    mu_raw: Dict[str, float],
    confidence: Dict[str, float],
    returns_hist: Dict[str, Sequence[float]],
    config,
    *,
    deployable_frac: Optional[float] = None,
    trading_days_per_year: Optional[int] = None,
    risk_reward: Optional[Dict[str, float]] = None,
) -> OptimizeResult:
    """Size ``buy_set`` to a max-Sharpe weight vector (fractions of equity).

    ``mu_raw`` = 15-day forecast return per name; ``returns_hist`` = aligned daily
    log-returns (<= cutoff) used for Σ and per-name vol. ``deployable_frac`` is the
    *upper bound* on deployment (defaults to ``1 - cash_floor_frac``).

    When ``risk_reward`` is supplied the *total* deployment is further scaled by the
    portfolio's conviction (:func:`deployment_fraction`): weak BUY sets deploy only
    a floor and keep cash; only excellent sets deploy the whole budget. The
    Markowitz solve still fixes the *relative* split among names; conviction fixes
    *how much* of the book to commit. Without ``risk_reward`` the full
    ``deployable_frac`` is deployed (legacy behaviour).
    """
    names = [t for t in buy_set if t in mu_raw and t in returns_hist]
    if deployable_frac is None:
        # Conviction mode may deploy up to 100% (exhaust) for excellent deals, so
        # the ceiling is 1.0 and the reserve is set by the conviction curve. Legacy
        # (no risk_reward) keeps the fixed cash-floor reserve.
        deployable_frac = 1.0 if risk_reward is not None else config.deployable_frac
    tdays = trading_days_per_year or getattr(config, "trading_days_per_year", 252)
    if not names:
        return OptimizeResult({}, {}, {}, {}, float(deployable_frac), 0.0, float("nan"))

    # Align return history to the shortest common length so Σ is well-formed.
    series = {t: np.asarray(returns_hist[t], dtype=np.float64) for t in names}
    min_len = min(len(v) for v in series.values())
    if min_len < 2:
        # Not enough history to estimate risk → equal-weight within caps.
        mu_vec = tilt_expected_returns(
            [mu_raw[t] for t in names], [confidence.get(t, 0.5) for t in names],
            config.llm_confidence_tilt,
        )
        caps = np.full(len(names), config.per_name_cap_frac)
        w0 = _renormalise(np.minimum(caps, deployable_frac / len(names)), caps, deployable_frac)
        w, deploy_frac, wrr = _apply_conviction(w0, caps, names, risk_reward, deployable_frac, config)
        w = apply_min_weight_floor(w, getattr(config, "min_weight_frac", 0.0))
        deploy_frac = float(w.sum())
        return OptimizeResult(
            {t: float(w[i]) for i, t in enumerate(names)},
            {t: float(mu_vec[i]) for i, t in enumerate(names)},
            {t: 0.0 for t in names},
            {t: float(caps[i]) for i, t in enumerate(names)},
            float(deployable_frac), deploy_frac, wrr,
        )

    matrix = np.column_stack([series[t][-min_len:] for t in names])
    cov_daily = shrunk_covariance(matrix)
    cov_annual = cov_daily * float(tdays)
    sigma_annual = np.sqrt(np.clip(np.diag(cov_annual), 0.0, None))

    mu_vec = tilt_expected_returns(
        [mu_raw[t] for t in names],
        [confidence.get(t, 0.5) for t in names],
        config.llm_confidence_tilt,
    )
    caps = per_name_caps(sigma_annual, config.per_name_cap_frac, config.vol_target_annual)
    rf_daily_15 = float(getattr(config, "risk_free_annual", 0.0)) * (15.0 / float(tdays))
    w0 = max_sharpe_weights(mu_vec, cov_annual, caps, deployable_frac, rf=rf_daily_15)
    weights, deploy_frac, wrr = _apply_conviction(w0, caps, names, risk_reward, deployable_frac, config)
    weights = apply_min_weight_floor(weights, getattr(config, "min_weight_frac", 0.0))
    deploy_frac = float(weights.sum())

    return OptimizeResult(
        weights={t: float(weights[i]) for i, t in enumerate(names)},
        expected_return={t: float(mu_vec[i]) for i, t in enumerate(names)},
        sigma_annual={t: float(sigma_annual[i]) for i, t in enumerate(names)},
        cap={t: float(caps[i]) for i, t in enumerate(names)},
        deployable_budget=float(deployable_frac),
        deploy_fraction=deploy_frac, weighted_risk_reward=wrr,
    )


def apply_min_weight_floor(weights, min_weight_frac: float) -> np.ndarray:
    """Zero out any allocation below ``min_weight_frac`` of the book.

    Positions the Markowitz solve wants to size below the minimum ticket weight
    are simply dropped (set to 0) rather than taken — the freed capital stays in
    cash, it is **not** redistributed across the surviving names. This enforces
    the "no investment below X% of capital; otherwise make it 0%" rule uniformly
    for both the backtest and production, since both go through this engine.
    """
    w = np.asarray(weights, dtype=np.float64)
    min_frac = float(min_weight_frac)
    if min_frac <= 0.0:
        return w
    return np.where(w < min_frac, 0.0, w)


def _apply_conviction(w0, caps, names, risk_reward, deployable_frac, config):
    """Scale the relative Markowitz weights ``w0`` to the conviction deployment.

    Returns ``(weights, deploy_fraction, weighted_risk_reward)``. With no per-name
    risk-reward the deployment equals ``deployable_frac`` (legacy, conviction=off).
    """
    w0 = np.asarray(w0, dtype=np.float64)
    total = float(w0.sum())
    if risk_reward is None or total <= 0:
        return w0, float(total), float("nan")
    rr_vec = np.array([float(risk_reward.get(t, 1.0)) for t in names], dtype=np.float64)
    weighted_rr = float(np.sum(w0 * rr_vec) / total)          # RR of the Sharpe-optimal mix
    deploy = deployment_fraction(weighted_rr, config)
    # Deploy the conviction fraction, but never above the caller's hard ceiling
    # (deployable_frac) or what the per-name caps can physically absorb.
    target = min(deploy, float(deployable_frac), float(caps.sum()))
    weights = _renormalise(w0 * (target / total), caps, target)
    return weights, float(weights.sum()), weighted_rr


# ---------------------------------------------------------------------------
# Cash-guard reconciliation (target notionals -> orders, never negative cash)
# ---------------------------------------------------------------------------

@dataclass
class PositionView:
    ticker: str
    qty: float
    price: float                 # current mark price (₹/share)
    risk_reward: float = 1.0     # incumbent's own last reward-to-risk
    action: str = "HOLD"         # last LLM call still governing this holding

    @property
    def notional(self) -> float:
        return float(self.qty) * float(self.price)


@dataclass
class BuyMeta:
    ticker: str
    price: float                 # entry price used for sizing (₹/share)
    risk_reward: float = 1.0
    confidence: float = 0.5


@dataclass
class Order:
    ticker: str
    side: str                    # "BUY" | "SELL"
    qty: float
    price: float
    notional: float
    reason: str
    funded_by: List[str] = field(default_factory=list)   # tickers rotated out to fund a BUY


@dataclass
class ReconcileResult:
    orders: List[Order]
    reallocations: List[Dict[str, object]]   # explicit "pulled X from A to fund B" records
    projected_cash: float                    # cash after all orders settle (>= 0)
    notes: List[str] = field(default_factory=list)


def reconcile_orders(
    *,
    cash: float,
    positions: Dict[str, PositionView],
    target_notional: Dict[str, float],       # BUY ticker -> desired ₹ (from optimizer)
    buy_meta: Dict[str, BuyMeta],
    avoid: Sequence[str],
    hold: Sequence[str],
    config,
    equity: Optional[float] = None,
    min_cash_reserve: Optional[float] = None,
) -> ReconcileResult:
    """Convert optimizer targets into orders under a hard no-negative-cash guard.

    Order of operations (all within one rebalance; in the walk-forward every order
    settles at the *next* bar open so a same-batch SELL funds a same-batch BUY):

    1. **AVOID** and dropped names (held but no longer wanted) → SELL to zero.
    2. **BUY trims** — a held BUY name above its target → SELL the excess.
    3. **BUY adds** — processed best-risk-reward first. Funded first from *free*
       cash (that above the ``cash_floor_frac * equity`` reserve). If that is not
       enough, capital is rotated out of the *weakest* eligible holding (one whose
       risk-reward the BUY beats by ``reallocation_edge``) — an asset-to-asset swap
       that does not touch the reserve. If still short, the BUY is downsized to
       what is actually fundable. Cash can never go negative and no funds are
       invented: the only money that moves is free cash and proceeds from selling
       shares already held.

    ``target_notional`` is the optimiser's *desired* ₹ per BUY (weights * equity).
    HOLD names are frozen — never trimmed or grown — and are only ever liquidated
    as a funding source of last resort when a strictly better BUY needs the capital.
    ``equity`` seeds the reserve floor; if omitted it defaults to ``cash`` (so the
    floor is simply a fraction of starting cash). ``min_cash_reserve`` overrides the
    reserve in rupees — the conviction-scaled callers pass ``(1 - deploy_fraction)
    * equity`` so a weak BUY set keeps more cash and an excellent one can deploy in
    full.
    """
    avoid_set = {t.upper() for t in avoid}
    hold_set = {t.upper() for t in hold}
    min_ticket = float(config.min_ticket_inr)
    edge = float(config.reallocation_edge)
    equity = float(cash) if equity is None else float(equity)
    if min_cash_reserve is not None:
        min_cash = max(0.0, float(min_cash_reserve))
    else:
        min_cash = float(getattr(config, "cash_floor_frac", 0.0)) * equity

    orders: List[Order] = []
    reallocations: List[Dict[str, object]] = []
    notes: List[str] = []
    available = float(cash)

    # Mutable view of remaining holdings we may still draw down for funding.
    remaining: Dict[str, float] = {t.upper(): float(p.qty) for t, p in positions.items()}
    pos_by_ticker = {t.upper(): p for t, p in positions.items()}

    def _sell(ticker: str, qty: float, reason: str) -> float:
        """Emit a SELL for ``qty`` shares, credit proceeds, return ₹ raised."""
        nonlocal available
        p = pos_by_ticker[ticker]
        qty = min(qty, remaining.get(ticker, 0.0))
        if qty <= 0 or p.price <= 0:
            return 0.0
        proceeds = qty * p.price
        remaining[ticker] = remaining.get(ticker, 0.0) - qty
        available += proceeds
        orders.append(Order(ticker, "SELL", qty, p.price, proceeds, reason))
        return proceeds

    # 1) Liquidate AVOID + names no longer targeted (not held-frozen, not a BUY).
    target_up = {t.upper(): float(v) for t, v in target_notional.items()}
    for ticker, p in pos_by_ticker.items():
        if ticker in avoid_set:
            _sell(ticker, remaining.get(ticker, 0.0), "AVOID: full exit")
        elif ticker not in target_up and ticker not in hold_set:
            _sell(ticker, remaining.get(ticker, 0.0), "dropped from BUY set: exit")

    # 2) Trim held BUY names that are above their new target.
    for ticker, tgt in sorted(target_up.items()):
        p = pos_by_ticker.get(ticker)
        if p is None or p.price <= 0:
            continue
        cur_notional = remaining.get(ticker, 0.0) * p.price
        if cur_notional - tgt > min_ticket:
            excess_qty = _whole_shares((cur_notional - tgt) / p.price, side="SELL")
            if excess_qty > 0:
                _sell(ticker, float(excess_qty), f"trim toward target ₹{tgt:,.0f}")

    # 3) Fund BUY adds, best risk-reward first.
    def _buy_rr(ticker: str) -> float:
        m = buy_meta.get(ticker)
        return m.risk_reward if m else 1.0

    for ticker in sorted(target_up, key=lambda t: (-_buy_rr(t), t)):
        tgt = target_up[ticker]
        meta = buy_meta.get(ticker)
        price = float(meta.price) if meta and meta.price and meta.price > 0 else (
            pos_by_ticker[ticker].price if ticker in pos_by_ticker else 0.0
        )
        if price <= 0 or tgt <= 0:
            continue
        held_notional = remaining.get(ticker, 0.0) * price
        need = tgt - held_notional
        if need <= min_ticket:
            continue

        # (a) Spend free cash — that which sits above the conviction reserve.
        free = max(0.0, available - min_cash)
        spend_from_cash = min(need, free)
        shortfall = need - spend_from_cash

        # (a2) Reallocation is a LAST RESORT: before rotating capital out of other
        #      holdings, use whatever cash is still on hand down to a minimal hard
        #      floor (``cash_floor_frac`` of equity). The conviction reserve set
        #      how much to *target*; it should not force a sale of another stock
        #      while cash sits idle. Only the shortfall that remains after using
        #      all deployable cash is funded by reallocation.
        if shortfall > min_ticket:
            hard_floor = min(min_cash, float(getattr(config, "cash_floor_frac", 0.0)) * equity)
            reserve_cash = max(0.0, available - spend_from_cash - hard_floor)
            dip = min(shortfall, reserve_cash)
            spend_from_cash += dip
            shortfall -= dip

        # (b) Rotate capital out of weaker incumbents to cover what's still short.
        #     Selling a holding credits `available`; spending exactly the raised
        #     amount back out is an asset-to-asset swap.
        raised = 0.0
        if shortfall > min_ticket:
            raised = _raise_via_reallocation(
                buyer=ticker, buyer_rr=_buy_rr(ticker), shortfall=shortfall,
                remaining=remaining, pos_by_ticker=pos_by_ticker, avoid_set=avoid_set,
                target_up=target_up, edge=edge, min_ticket=min_ticket, sell=_sell,
                reallocations=reallocations, buyer_price=price,
            )

        spend = spend_from_cash + min(shortfall, raised)
        spend = min(spend, available)                 # hard non-negative guard
        if spend < min_ticket:
            notes.append(f"{ticker}: BUY skipped — only ₹{spend:,.0f} fundable (< ticket)")
            continue
        # Whole shares only: buy as many as `spend` affords (floor), then the
        # actual cash outlay is exactly qty*price — any remainder stays as cash.
        qty = _whole_shares(spend / price, side="BUY")
        if qty < 1:
            notes.append(f"{ticker}: BUY skipped — ₹{spend:,.0f} buys <1 whole share at ₹{price:,.0f}")
            continue
        spend = qty * price
        available -= spend
        funded = sorted({r["from"] for r in reallocations if r["to"] == ticker})  # type: ignore[index]
        reason = "BUY to target" if spend >= need - 1e-6 else "BUY (capped by fundable capital)"
        orders.append(Order(ticker, "BUY", float(qty), price, spend, reason, funded_by=funded))

    available = max(0.0, available)
    return ReconcileResult(orders, reallocations, float(available), notes)


def _raise_via_reallocation(
    *,
    buyer: str,
    buyer_rr: float,
    shortfall: float,
    remaining: Dict[str, float],
    pos_by_ticker: Dict[str, PositionView],
    avoid_set,
    target_up: Dict[str, float],
    edge: float,
    min_ticket: float,
    sell,
    reallocations: List[Dict[str, object]],
    buyer_price: float,
) -> float:
    """Sell down the weakest eligible holdings to raise ``shortfall`` for ``buyer``.

    An incumbent is *eligible* only if the buyer beats it on risk-reward by the
    configured edge (``buyer_rr > incumbent_rr * (1 + edge)``) and the incumbent is
    not itself a name we are trying to buy up to target. Weakest (lowest
    risk-reward) incumbents are drained first. This is the explicit realisation of
    "if the suggested stock looks better, pull the money out of an existing one" —
    and because we only ever sell shares we already hold, no new funds are created.
    Returns the total ₹ raised.
    """
    candidates = []
    for ticker, p in pos_by_ticker.items():
        if ticker == buyer or ticker in avoid_set:
            continue
        held_value = remaining.get(ticker, 0.0) * p.price
        if held_value < min_ticket or p.price <= 0:
            continue
        # A name being bought *up* to target should not be raided to fund another.
        already_target = target_up.get(ticker, 0.0) > held_value + min_ticket
        if already_target:
            continue
        if buyer_rr > p.risk_reward * (1.0 + edge):
            candidates.append((p.risk_reward, ticker, held_value))

    candidates.sort()  # weakest risk-reward first
    raised = 0.0
    for incumbent_rr, ticker, held_value in candidates:
        if raised >= shortfall - 1e-6:
            break
        want = shortfall - raised
        p = pos_by_ticker[ticker]
        # Whole shares only: raise at least `want` (round up), never exceeding
        # the shares held. The remaining book is integer, so this stays integer.
        held = remaining.get(ticker, 0.0)
        sell_qty = min(held, float(_whole_shares(want / p.price, side="SELL")))
        if sell_qty <= 0:
            continue
        proceeds = sell(ticker, sell_qty, f"reallocate → {buyer} (RR {buyer_rr:.2f} > {incumbent_rr:.2f})")
        if proceeds > 0:
            raised += proceeds
            reallocations.append({
                "from": ticker,
                "to": buyer,
                "amount": round(proceeds, 2),
                "from_risk_reward": round(float(incumbent_rr), 3),
                "to_risk_reward": round(float(buyer_rr), 3),
                "edge": edge,
            })
    return raised
