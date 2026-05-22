"""Archive-as-inbox retry queue for Firestore uploads.

Folder convention:

* Upload staging data lives at ``Data/archive/<name>.parquet``.
* There is no persistent ``Data/main`` cache for Firestore-bound ingest
  outputs. Successful uploads delete the staged file; failures retain it for
  the next run.

Behavior contract (fall-through variant — collection enabled):

1. **Drain step** — if a parquet exists at the archive path, try to upload
   and delete it. On failure the file is retained for the next attempt.

2. **Fresh-collect step** — always runs, regardless of whether the drain
   succeeded.

3. **Merge-stage step** — the new collected data is *merged* into the
   archive file (which may still contain rows from a failed drain) with
   dedup on the caller-supplied keys. This guarantees no data loss when
   Firestore is unreachable across multiple runs.

4. **Upload-and-clear step** — try to upload the (possibly merged) archive.
   Delete it on success, retain it on failure for the next run.

In steady state (Firestore healthy) the archive file lives for milliseconds
between stage and clear. Only failed uploads leave a file behind.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is part of runtime deps
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def truthy_env(name: str, default: bool = True) -> bool:
    """Parse a per-module ``COLLECT_*`` env var.

    Accepts ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive) as truthy.
    Empty or unset → ``default``. Anything else → ``False``.
    Use this to gate the fresh-collection branch in each ingestion module.
    """
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def archive_path_for(path: Path) -> Path:
    """Return the Firestore staging path for ``path``.

    Callers now pass archive paths directly. For older flat paths such as
    ``Data/news.parquet``, keep mapping to ``Data/archive/news.parquet``.
    """
    if path.parent.name == "archive":
        return path
    return path.parent / "archive" / path.name


def has_pending_archive(archive_path: Path) -> bool:
    """True iff a non-empty archive parquet exists at ``archive_path``."""
    return archive_path.exists() and archive_path.stat().st_size > 0


def drain_pending_archive(
    archive_path: Path,
    upload_fn: Callable[[pd.DataFrame], int],
    log: Callable[[str], None],
) -> Optional[int]:
    """Try to upload the pending archive parquet and delete it on success.

    Returns:
        - ``None`` if no archive file exists (caller should proceed with
          fresh collection).
        - An integer row count if a file existed. ``0`` means the file
          existed but either was empty or the upload failed (the caller
          should still treat this as "archive was pending" and skip fresh
          collection per the spec).

    On upload failure the archive file is retained for the next run.
    """
    if not has_pending_archive(archive_path):
        return None
    try:
        pending = pd.read_parquet(archive_path)
    except Exception as exc:
        log(f"[archive] failed to read {archive_path}: {exc}; retaining for next run")
        return 0
    if pending.empty:
        archive_path.unlink(missing_ok=True)
        log(f"[archive] {archive_path} was empty; removed")
        return 0
    log(f"[archive] draining {archive_path} ({len(pending)} rows)")
    try:
        written = int(upload_fn(pending))
    except Exception as exc:
        log(f"[archive] upload of {archive_path} failed: {exc}; retaining for next run")
        return 0
    if written < len(pending):
        log(
            f"[archive] upload of {archive_path} wrote {written}/{len(pending)} rows; "
            "retaining for retry"
        )
        return 0
    archive_path.unlink(missing_ok=True)
    log(f"[archive] uploaded {written} rows from {archive_path.name}; file removed")
    return written


def stage_to_archive(
    df: pd.DataFrame,
    archive_path: Path,
    log: Callable[[str], None],
) -> bool:
    """Write ``df`` to the archive parquet (overwrites any existing file).

    Returns True if a file was written, False if the frame was empty.
    """
    if df is None or df.empty:
        return False
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(archive_path, index=False)
    log(f"[archive] staged {len(df)} rows to {archive_path}")
    return True


def merge_stage_to_archive(
    df: pd.DataFrame,
    archive_path: Path,
    dedup_keys: Optional[Iterable[str]],
    log: Callable[[str], None],
) -> int:
    """Stage ``df`` to the archive, **merging** with any retained rows.

    Used on the fall-through path: after a (possibly failed) drain, the
    archive may still contain rows from yesterday's failed upload. We don't
    want today's fresh data to clobber those — instead we concatenate and
    dedup on ``dedup_keys`` (keep="last", so today's row wins per key).

    Returns the row count actually written. Returns 0 when both the input
    and the retained archive are empty.
    """
    if df is None:
        df = pd.DataFrame()

    if archive_path.exists():
        try:
            retained = pd.read_parquet(archive_path)
            if not retained.empty:
                log(f"[archive] merging {len(df)} new rows with {len(retained)} retained")
                combined = pd.concat([retained, df], ignore_index=True) if not df.empty else retained
            else:
                combined = df
        except Exception as exc:
            log(f"[archive] failed to read retained {archive_path}: {exc}; overwriting")
            combined = df
    else:
        combined = df

    if combined is None or combined.empty:
        return 0

    if dedup_keys:
        keys = [k for k in dedup_keys if k in combined.columns]
        if keys:
            combined = combined.drop_duplicates(subset=keys, keep="last")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(archive_path, index=False)
    log(f"[archive] staged {len(combined)} rows to {archive_path}")
    return len(combined)


def upload_and_clear_archive(
    archive_path: Path,
    upload_fn: Callable[[pd.DataFrame], int],
    log: Callable[[str], None],
) -> int:
    """Read ``archive_path``, upload via ``upload_fn``, delete on success.

    Used on the case-3 path (fresh collect → archive staged → upload).
    Returns rows uploaded. On upload failure the archive is retained so the
    next run can drain it via :func:`drain_pending_archive`.
    """
    if not has_pending_archive(archive_path):
        log("[archive] no archive file to upload")
        return 0
    df = pd.read_parquet(archive_path)
    if df.empty:
        archive_path.unlink(missing_ok=True)
        return 0
    try:
        written = int(upload_fn(df))
    except Exception as exc:
        log(f"[archive] upload of {archive_path} failed: {exc}; retaining for retry")
        return 0
    if written < len(df):
        log(
            f"[archive] upload of {archive_path} wrote {written}/{len(df)} rows; "
            "retaining for retry"
        )
        return 0
    archive_path.unlink(missing_ok=True)
    log(f"[archive] uploaded {written} rows; {archive_path.name} removed")
    return written
