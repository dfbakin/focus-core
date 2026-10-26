"""Tests for session storage and the video source setting."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.core.video_source import CAMERA, FILE, VideoSource
from app.db.local import init_db
from app.db.sessions import (
    create_session,
    delete_sessions,
    finish_session,
    list_sessions,
    load_engagement_points,
    load_session_report,
    save_engagement_points,
    save_engagement_summary,
)
from app.db.settings import (
    load_min_free_ram_mb,
    load_simulated_error_rate,
    load_video_source,
    save_min_free_ram_mb,
    save_simulated_error_rate,
    save_video_source,
)

START = datetime(2026, 9, 26, 10, 0)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _finished_session(db_path: Path, started: datetime, minutes: int) -> str:
    session_id = create_session(CAMERA, 40, db_path, started_at=started)
    finish_session(session_id, db_path, finished_at=started + timedelta(minutes=minutes))
    return session_id


def test_unfinished_session_is_not_listed(db_path: Path) -> None:
    create_session(CAMERA, 40, db_path, started_at=START)
    assert list_sessions(db_path) == []


def test_finished_session_has_duration(db_path: Path) -> None:
    _finished_session(db_path, START, minutes=45)
    [session] = list_sessions(db_path)
    assert session.duration_sec == 45 * 60
    assert session.avg_engagement is None


def test_sessions_are_listed_newest_first(db_path: Path) -> None:
    _finished_session(db_path, START, 10)
    newer = _finished_session(db_path, START + timedelta(days=1), 10)
    assert list_sessions(db_path)[0].id == newer


def test_engagement_summary_is_stored(db_path: Path) -> None:
    session_id = _finished_session(db_path, START, 10)
    save_engagement_summary(session_id, 62.5, 30.0, db_path)
    assert list_sessions(db_path)[0].avg_engagement == 62.5


def test_delete_removes_only_selected(db_path: Path) -> None:
    first = _finished_session(db_path, START, 10)
    second = _finished_session(db_path, START + timedelta(days=1), 10)
    delete_sessions([first], db_path)
    assert [s.id for s in list_sessions(db_path)] == [second]


def test_video_source_round_trip(db_path: Path) -> None:
    assert load_video_source(db_path) is None
    camera = VideoSource(CAMERA, camera_index=1201, camera_name="FaceTime HD Camera")
    save_video_source(camera, db_path)
    assert load_video_source(db_path) == camera
    save_video_source(VideoSource(FILE, file_path="/tmp/a.mp4"), db_path)
    assert load_video_source(db_path) == VideoSource(FILE, file_path="/tmp/a.mp4")


def test_min_free_ram_default_and_override(db_path: Path) -> None:
    assert load_min_free_ram_mb(db_path) == 1024
    save_min_free_ram_mb(64000, db_path)
    assert load_min_free_ram_mb(db_path) == 64000


def test_explicit_duration_excludes_pauses(db_path: Path) -> None:
    session_id = create_session(CAMERA, 40, db_path, started_at=START)
    finish_session(
        session_id, db_path, finished_at=START + timedelta(hours=2), duration_sec=3000
    )
    assert list_sessions(db_path)[0].duration_sec == 3000


def test_engagement_points_round_trip_and_cascade(db_path: Path) -> None:
    session_id = _finished_session(db_path, START, 10)
    save_engagement_points(session_id, [(10, 55.0), (5, 70.0)], db_path)
    assert load_engagement_points(session_id, db_path) == [(5, 70.0), (10, 55.0)]
    delete_sessions([session_id], db_path)
    assert load_engagement_points(session_id, db_path) == []


def test_simulated_error_rate_is_off_by_default(db_path: Path) -> None:
    assert load_simulated_error_rate(db_path) == 0.0
    save_simulated_error_rate(0.3, db_path)
    assert load_simulated_error_rate(db_path) == 0.3


def test_session_report_is_assembled(db_path: Path) -> None:
    session_id = _finished_session(db_path, START, 10)
    save_engagement_points(session_id, [(0, 80.0), (5, 20.0)], db_path)
    report = load_session_report(session_id, db_path)
    assert report is not None
    assert report.threshold_pct == 40
    assert report.minimum == 20.0
    assert load_session_report("missing", db_path) is None
