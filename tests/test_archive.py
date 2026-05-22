"""Tests for ingestion/_archive.py — the retry-queue helpers.

Behaviour contract (from the user spec):
* Pending archive parquet exists → upload + delete on success; retain on failure.
* No pending archive → caller proceeds to fresh-collect, stages a new file,
  uploads it, deletes on success / keeps on failure.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from ingestion import _archive


def _log_to(lines):
    return lambda msg: lines.append(msg)


def _sample_df():
    return pd.DataFrame(
        [
            {"ticker": "RELIANCE", "value": 1},
            {"ticker": "TCS", "value": 2},
        ]
    )


# ----------------------------------------------------------------------------
# archive_path_for + has_pending_archive
# ----------------------------------------------------------------------------


def test_archive_path_for_uses_sibling_archive_folder(tmp_path):
    main = tmp_path / "news.parquet"
    assert _archive.archive_path_for(main) == tmp_path / "archive" / "news.parquet"


def test_has_pending_archive_false_for_missing_file(tmp_path):
    assert _archive.has_pending_archive(tmp_path / "archive" / "x.parquet") is False


def test_has_pending_archive_false_for_zero_byte_file(tmp_path):
    p = tmp_path / "archive" / "x.parquet"
    p.parent.mkdir(parents=True)
    p.touch()
    assert _archive.has_pending_archive(p) is False


def test_has_pending_archive_true_for_non_empty(tmp_path):
    p = tmp_path / "archive" / "x.parquet"
    p.parent.mkdir(parents=True)
    _sample_df().to_parquet(p, index=False)
    assert _archive.has_pending_archive(p) is True


# ----------------------------------------------------------------------------
# drain_pending_archive
# ----------------------------------------------------------------------------


def test_drain_returns_none_when_no_archive(tmp_path):
    p = tmp_path / "archive" / "x.parquet"
    logs = []
    result = _archive.drain_pending_archive(p, upload_fn=lambda df: len(df), log=_log_to(logs))
    assert result is None
    assert logs == []


def test_drain_uploads_and_deletes_on_success(tmp_path):
    p = tmp_path / "archive" / "news.parquet"
    p.parent.mkdir(parents=True)
    _sample_df().to_parquet(p, index=False)
    logs = []
    result = _archive.drain_pending_archive(p, upload_fn=lambda df: len(df), log=_log_to(logs))
    assert result == 2
    assert not p.exists()  # archive cleared after successful upload


def test_drain_retains_file_when_upload_raises(tmp_path):
    p = tmp_path / "archive" / "news.parquet"
    p.parent.mkdir(parents=True)
    _sample_df().to_parquet(p, index=False)
    logs = []

    def failing_upload(df):
        raise RuntimeError("firestore unreachable")

    result = _archive.drain_pending_archive(p, upload_fn=failing_upload, log=_log_to(logs))
    assert result == 0  # 0 == "tried and failed"
    assert p.exists()  # retained for next-run retry
    assert any("retain" in msg.lower() for msg in logs)


def test_drain_retains_file_when_upload_writes_fewer_rows(tmp_path):
    p = tmp_path / "archive" / "news.parquet"
    p.parent.mkdir(parents=True)
    _sample_df().to_parquet(p, index=False)
    logs = []
    result = _archive.drain_pending_archive(p, upload_fn=lambda df: 1, log=_log_to(logs))
    assert result == 0
    assert p.exists()
    assert any("1/2" in msg for msg in logs)


def test_drain_clears_empty_archive_file(tmp_path):
    p = tmp_path / "archive" / "news.parquet"
    p.parent.mkdir(parents=True)
    pd.DataFrame(columns=["ticker"]).to_parquet(p, index=False)
    logs = []
    result = _archive.drain_pending_archive(p, upload_fn=lambda df: len(df), log=_log_to(logs))
    assert result == 0
    assert not p.exists()  # empty archive is removed


# ----------------------------------------------------------------------------
# stage_to_archive + upload_and_clear_archive
# ----------------------------------------------------------------------------


def test_stage_writes_parquet_and_returns_true(tmp_path):
    p = tmp_path / "archive" / "news.parquet"
    logs = []
    wrote = _archive.stage_to_archive(_sample_df(), p, log=_log_to(logs))
    assert wrote is True
    assert p.exists()
    roundtrip = pd.read_parquet(p)
    assert len(roundtrip) == 2


def test_stage_noop_for_empty_df(tmp_path):
    p = tmp_path / "archive" / "news.parquet"
    wrote = _archive.stage_to_archive(pd.DataFrame(), p, log=lambda m: None)
    assert wrote is False
    assert not p.exists()


def test_upload_and_clear_removes_file_on_success(tmp_path):
    p = tmp_path / "archive" / "news.parquet"
    p.parent.mkdir(parents=True)
    _sample_df().to_parquet(p, index=False)
    logs = []
    written = _archive.upload_and_clear_archive(p, upload_fn=lambda df: len(df), log=_log_to(logs))
    assert written == 2
    assert not p.exists()


def test_upload_and_clear_retains_file_on_failure(tmp_path):
    p = tmp_path / "archive" / "news.parquet"
    p.parent.mkdir(parents=True)
    _sample_df().to_parquet(p, index=False)
    logs = []

    def boom(df):
        raise RuntimeError("upload broke")

    written = _archive.upload_and_clear_archive(p, upload_fn=boom, log=_log_to(logs))
    assert written == 0
    assert p.exists()  # retained


def test_upload_and_clear_retains_file_when_upload_writes_fewer_rows(tmp_path):
    p = tmp_path / "archive" / "news.parquet"
    p.parent.mkdir(parents=True)
    _sample_df().to_parquet(p, index=False)
    logs = []
    written = _archive.upload_and_clear_archive(p, upload_fn=lambda df: 1, log=_log_to(logs))
    assert written == 0
    assert p.exists()
    assert any("1/2" in msg for msg in logs)


def test_upload_and_clear_noop_when_no_archive(tmp_path):
    p = tmp_path / "archive" / "missing.parquet"
    written = _archive.upload_and_clear_archive(p, upload_fn=lambda df: len(df), log=lambda m: None)
    assert written == 0
