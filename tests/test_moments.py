"""Tests for storing transcripts and comments of critical moments."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from app.core.report import CriticalMoment
from app.core.report_pdf import export_report_pdf
from app.core.video_source import CAMERA
from app.db.local import init_db
from app.db.moments import load_moment_notes, save_comment, save_transcripts
from app.db.sessions import (
    create_session,
    finish_session,
    load_session_report,
    save_engagement_points,
)

START = datetime(2026, 10, 28, 14, 0)
POINTS = [(0.0, 80.0), (10.5, 20.0), (21.0, 70.0), (31.5, 25.0), (42.0, 90.0)]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


@pytest.fixture
def session_id(db_path: Path) -> str:
    session_id = create_session(CAMERA, 40, db_path, started_at=START)
    finish_session(session_id, db_path)
    save_engagement_points(session_id, POINTS, db_path)
    return session_id


def test_transcripts_and_comments_reach_the_report(db_path: Path, session_id: str) -> None:
    moments = load_session_report(session_id, db_path).critical_moments
    save_transcripts(session_id, moments, ["первая расшифровка", None], db_path)
    save_comment(session_id, moments[0], "Добавить слайд", db_path)
    save_comment(session_id, moments[1], "Слишком быстро", db_path)

    report = load_session_report(session_id, db_path)
    first, second = (report.note_for(moment) for moment in report.critical_moments)
    assert (first.transcript, first.comment) == ("первая расшифровка", "Добавить слайд")
    assert (second.transcript, second.comment) == (None, "Слишком быстро")


def test_comment_is_replaced_and_removed(db_path: Path, session_id: str) -> None:
    moment = CriticalMoment(10.5, 21.0, 20.0)
    save_comment(session_id, moment, "черновик", db_path)
    save_comment(session_id, moment, "итог", db_path)
    assert load_moment_notes(session_id, db_path)[10.5].comment == "итог"
    save_comment(session_id, moment, "   ", db_path)
    assert load_moment_notes(session_id, db_path)[10.5].comment is None


def test_old_database_gets_the_transcript_column(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE critical_moments (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "start_sec INTEGER NOT NULL, end_sec INTEGER NOT NULL, min_value REAL, "
            "comment TEXT, synced_at TEXT)"
        )
    init_db(path)
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(critical_moments)")}
    assert "transcript" in columns


def test_long_transcripts_continue_on_next_pages(
    db_path: Path, session_id: str, tmp_path: Path
) -> None:
    moments = load_session_report(session_id, db_path).critical_moments
    long_text = "Рассмотрим формальное определение производной функции в точке. " * 40
    save_transcripts(session_id, moments, [long_text, long_text], db_path)
    path = tmp_path / "report.pdf"
    export_report_pdf(load_session_report(session_id, db_path), path)
    content = path.read_bytes()
    assert content.startswith(b"%PDF")
    assert content.count(b"/Type /Page\n") + content.count(b"/Type /Page ") >= 2
