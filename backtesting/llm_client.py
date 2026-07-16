"""Rotating multi-key / multi-provider LLM client for the backtest.

The 2020→2026 walk-forward fires tens of thousands of LLM calls; a single API
key hits rate / daily-token limits long before the run finishes. This client
holds an **ordered pool of providers** — the Ollama Cloud keys first, then the
Groq keys, all serving the *same* ``gpt-oss-120b`` model — and rotates through
them so the simulation keeps moving:

* It **sticks** with the current provider while it works (no needless switching).
* On **any** error from a provider (rate limit, auth, token/context overflow,
  network), it advances to the next provider and retries the *same* request.
* Order is Ollama ``key1..keyN`` → Groq ``key1..keyM``. **After the last Groq
  key it wraps back to Ollama ``key1``** — exactly the user's spec: "after the
  last Groq key, try again from the first Ollama key".
* A rate-limited/errored provider is put on a short **cooldown** so concurrent
  and subsequent calls skip it and spread load across the remaining keys.
* A single request fails only if *every* provider fails within one full cycle;
  ``run_signals`` then records that ticker's error for the bar and continues
  (non-fatal — the backtest is never aborted by one LLM hiccup).

The client is a drop-in for ``features.trade_suggestions.ask_glm``: it has the
same call signature, so it slots straight into ``run_signals(ask=...)``. It is
thread-safe (``run_signals`` fans out across a thread pool).
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from features.trade_suggestions import ask_glm

# Public OpenAI-compatible endpoints. Both serve gpt-oss-120b.
OLLAMA_BASE_URL = "https://ollama.com/v1"
OLLAMA_MODEL = "gpt-oss:120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"


@dataclass
class Provider:
    """One rotatable endpoint = base URL + key + model id, with a cooldown clock."""
    name: str
    base_url: str
    api_key: str
    model: str
    cooldown_until: float = field(default=0.0)


def _looks_rate_limited(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in ("429", "rate", "too many", "quota", "limit", "exceed"))


class RotatingLLMClient:
    """Ordered pool of LLM providers with sticky selection + failure rotation.

    Parameters
    ----------
    providers:
        Ordered list — Ollama keys first, then Groq keys. Cycled in order,
        wrapping from the last back to the first.
    cooldown_seconds:
        How long to bench a provider after a rate-limit-style error so load
        spreads to the other keys.
    max_cycles:
        How many full passes over the pool to attempt for one request before
        giving up (each provider is tried ``max_cycles`` times at most).
    per_call_attempts:
        Attempts *within* a single provider before rotating (1 = fail fast, the
        default — rotating to a fresh key is better than retrying a dead one).
    """

    def __init__(
        self,
        providers: List[Provider],
        *,
        cooldown_seconds: float = 60.0,
        max_cycles: int = 2,
        per_call_attempts: int = 1,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not providers:
            raise ValueError("RotatingLLMClient needs at least one provider")
        self.providers = providers
        self.cooldown_seconds = cooldown_seconds
        self.max_cycles = max(1, max_cycles)
        self.per_call_attempts = max(1, per_call_attempts)
        self._log = logger or (lambda m: print(m, file=sys.stderr))
        self._idx = 0
        self._lock = threading.Lock()

    # ── provider selection ────────────────────────────────────────────────
    def _pick(self, now: float) -> int:
        """Sticky index, skipping providers still cooling; least-cool if all are."""
        n = len(self.providers)
        with self._lock:
            order = [(self._idx + k) % n for k in range(n)]
            for i in order:
                if self.providers[i].cooldown_until <= now:
                    return i
            # everyone is cooling — take the one that recovers soonest.
            return min(range(n), key=lambda i: self.providers[i].cooldown_until)

    def _on_success(self, i: int) -> None:
        with self._lock:
            self._idx = i  # stick with the working key

    def _on_failure(self, i: int, exc: Exception, now: float) -> None:
        with self._lock:
            if _looks_rate_limited(exc):
                self.providers[i].cooldown_until = now + self.cooldown_seconds
            # move the sticky pointer forward so the next call starts elsewhere.
            if self._idx == i:
                self._idx = (i + 1) % len(self.providers)

    # ── ask_glm-compatible entrypoint ─────────────────────────────────────
    def __call__(
        self,
        ticker: str,
        market: Dict[str, Any],
        sentiment: Optional[Dict[str, Any]],
        model: str,          # ignored — each provider carries its own model id
        api_key: str,        # ignored — each provider carries its own key
        fundamentals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        n = len(self.providers)
        last_exc: Optional[Exception] = None
        for _ in range(n * self.max_cycles):
            now = time.monotonic()
            i = self._pick(now)
            prov = self.providers[i]
            try:
                result = ask_glm(
                    ticker, market, sentiment, prov.model, prov.api_key, fundamentals,
                    base_url=prov.base_url, max_attempts=self.per_call_attempts,
                )
                self._on_success(i)
                return result
            except Exception as exc:  # noqa: BLE001 — rotate on any provider error
                last_exc = exc
                self._log(f"[backtest-llm] {prov.name} failed for {ticker}: {exc} — rotating")
                self._on_failure(i, exc, now)
        raise RuntimeError(
            f"all {n} LLM providers exhausted for {ticker} "
            f"(last error: {last_exc})"
        )


# ---------------------------------------------------------------------------
# Construction from settings / env
# ---------------------------------------------------------------------------

def build_providers(
    ollama_keys: List[str],
    groq_keys: List[str],
    *,
    ollama_base_url: str = OLLAMA_BASE_URL,
    ollama_model: str = OLLAMA_MODEL,
    groq_base_url: str = GROQ_BASE_URL,
    groq_model: str = GROQ_MODEL,
) -> List[Provider]:
    """Ollama providers first (in order), then Groq providers (in order)."""
    providers: List[Provider] = []
    for n, key in enumerate(ollama_keys, 1):
        if key.strip():
            providers.append(Provider(f"ollama#{n}", ollama_base_url, key.strip(), ollama_model))
    for n, key in enumerate(groq_keys, 1):
        if key.strip():
            providers.append(Provider(f"groq#{n}", groq_base_url, key.strip(), groq_model))
    return providers


def from_settings(settings: Any, *, logger: Optional[Callable[[str], None]] = None) -> Optional["RotatingLLMClient"]:
    """Build a client from a ``BacktestSettings`` (or ``None`` if no keys set)."""
    providers = build_providers(
        list(getattr(settings, "ollama_api_keys", []) or []),
        list(getattr(settings, "groq_api_keys", []) or []),
        ollama_base_url=getattr(settings, "ollama_base_url", OLLAMA_BASE_URL),
        ollama_model=getattr(settings, "ollama_model", OLLAMA_MODEL),
        groq_base_url=getattr(settings, "groq_base_url", GROQ_BASE_URL),
        groq_model=getattr(settings, "groq_model", GROQ_MODEL),
    )
    if not providers:
        return None
    return RotatingLLMClient(providers, logger=logger)
