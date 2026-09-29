"""Tests for the retention policy and storage cleanup."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.core.retention import RetentionPolicy
from app.core.video_source import CAMERA
from app.db.local import init_db
from app.db.maintenance import (
    apply_retention_policy,
    delete_expired_sessions,
    storage_status,
)
from app.db.sessions import create_session, finish_session, list_sessions
from app.db.settings import load_retention_policy, save_retention_policy

NOW = datetime(2026, 9, 28, 12, 0)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _session_days_ago(db_path: Path, days: int) -> str:
    started = NOW - timedelta(days=days)
    session_id = create_session(CAMERA, 40, db_path, started_at=started)
    finish_session(session_id, db_path, finished_at=started + timedelta(minutes=45))
    return session_id


def test_only_expired_sessions_are_deleted(db_path: Path) -> None:
    _session_days_ago(db_path, 40)
    recent = _session_days_ago(db_path, 5)
    result = delete_expired_sessions(30, db_path, now=NOW)
    assert result.deleted_count == 1
    assert [s.id for s in list_sessions(db_path)] == [recent]


def test_nothing_to_delete(db_path: Path) -> None:
    _session_days_ago(db_path, 2)
    result = delete_expired_sessions(30, db_path, now=NOW)
    assert result.deleted_count == 0
    assert result.freed_bytes == 0


def test_storage_status_counts_finished_sessions(db_path: Path) -> None:
    _session_days_ago(db_path, 1)
    create_session(CAMERA, 40, db_path)
    status = storage_status(db_path)
    assert status.session_count == 1
    assert status.size_bytes > 0


def test_policy_round_trip_and_default(db_path: Path) -> None:
    assert load_retention_policy(db_path) == RetentionPolicy()
    policy = RetentionPolicy(auto_delete=True, days=14)
    save_retention_policy(policy, db_path)
    assert load_retention_policy(db_path) == policy


def test_disabled_policy_deletes_nothing(db_path: Path) -> None:
    _session_days_ago(db_path, 400)
    assert apply_retention_policy(RetentionPolicy(auto_delete=False), db_path) is None
    assert len(list_sessions(db_path)) == 1
