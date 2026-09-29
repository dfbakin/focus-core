"""Tests for storing analysis settings in SQLite."""

from pathlib import Path

import pytest

from app.core.config import AnalysisSettings
from app.db.local import get_connection, init_db
from app.db.settings import load_settings, save_settings


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


def test_empty_database_returns_defaults(db_path: Path) -> None:
    assert load_settings(db_path) == AnalysisSettings()


def test_saved_settings_are_loaded_back(db_path: Path) -> None:
    settings = AnalysisSettings(threshold_pct=25, interval_sec=2.5, smoothing_window=7)
    save_settings(settings, db_path)
    assert load_settings(db_path) == settings


def test_second_save_overwrites_first(db_path: Path) -> None:
    save_settings(AnalysisSettings(threshold_pct=10), db_path)
    save_settings(AnalysisSettings(threshold_pct=90), db_path)
    assert load_settings(db_path).threshold_pct == 90


def test_malformed_value_falls_back_to_default(db_path: Path) -> None:
    with get_connection(db_path) as conn:
        conn.execute("INSERT INTO settings (key, value) VALUES ('threshold_pct', 'abc')")
    assert load_settings(db_path).threshold_pct == AnalysisSettings().threshold_pct
