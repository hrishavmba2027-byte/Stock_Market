"""Point-in-time (PIT) data spine — append-only, as-of-dated archives.

Phase 3. The live Firestore collections keep only the *latest* snapshot (by
design — replace semantics). To backtest without look-ahead we also need to
know what each signal looked like on a *past* date. This module maintains a
**parallel, append-only** parquet store under ``Settings.pit_archive_dir``:

* Every ingest run appends its rows; nothing is ever removed.
* Deduplication is on ``(entity keys + snapshot/as-of column)`` so re-running
  the same day is idempotent, but a later day's snapshot accumulates alongside
  the earlier ones (revisions stay visible).

It is entirely gated by ``Settings.enable_pit_archive`` (env
``ENABLE_PIT_ARCHIVE``); when off, :func:`maybe_append_pit` is a cheap no-op and
the production replace-only paths behave exactly as before.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd


def _default_log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _settings() -> Any:
    from app.config.settings import get_settings

    return get_settings()


def is_pit_enabled() -> bool:
    try:
        return bool(_settings().enable_pit_archive)
    except Exception:  # pragma: no cover - config failure ⇒ PIT off
        return False


def pit_path(name: str, *, base_dir: Optional[Path] = None) -> Path:
    if base_dir is None:
        base_dir = _settings().pit_archive_dir
    return Path(base_dir) / f"{name}.parquet"


def append_pit(
    df: pd.DataFrame,
    name: str,
    dedup_keys: Iterable[str],
    *,
    path: Optional[Path] = None,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """Append ``df`` to the PIT parquet ``name``, never removing prior rows.

    Returns the total row count after the merge. Dedup keeps the last row per
    ``dedup_keys`` group so re-ingesting the same as-of date overwrites that
    snapshot in place while distinct dates accumulate.
    """
    log = log or _default_log
    if df is None or df.empty:
        return 0
    if path is None:
        path = pit_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)

    incoming = df.copy()
    if path.exists():
        try:
            existing = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - corrupt file guard
            log(f"[pit] failed to read {path}: {exc}; rewriting from incoming only")
            existing = pd.DataFrame()
        combined = (
            pd.concat([existing, incoming], ignore_index=True)
            if not existing.empty
            else incoming
        )
    else:
        combined = incoming

    keys = [k for k in dedup_keys if k in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")

    combined.to_parquet(path, index=False)
    log(f"[pit] {name}: {len(combined)} rows total (append-only) → {path}")
    return len(combined)


def maybe_append_pit(
    df: pd.DataFrame,
    name: str,
    dedup_keys: Iterable[str],
    *,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """Append to the PIT store only when ``ENABLE_PIT_ARCHIVE`` is on."""
    if not is_pit_enabled():
        return 0
    if df is None or df.empty:
        return 0
    try:
        return append_pit(df, name, dedup_keys, log=log)
    except Exception as exc:  # pragma: no cover - PIT must never break ingest
        (log or _default_log)(f"[pit] append for {name} failed (non-fatal): {exc}")
        return 0


def load_pit(name: str, *, path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = pit_path(name)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_pit_asof(
    name: str,
    asof: Any,
    date_col: str,
    *,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return the slice of PIT archive ``name`` visible on ``asof`` (inclusive).

    Rows whose ``date_col`` is after ``asof`` are dropped — this is the
    look-ahead guard the backtest relies on.
    """
    df = load_pit(name, path=path)
    if df.empty or date_col not in df.columns:
        return df
    cutoff = pd.to_datetime(asof, utc=True, errors="coerce")
    col = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    return df[col <= cutoff].reset_index(drop=True)
