"""Tests for report calculations and PDF export."""

from datetime import datetime
from pathlib import Path

import pytest

from app.core.report import (
    CriticalMoment,
    MomentNote,
    SessionReport,
    find_critical_moments,
    find_dips,
)
from app.core.report_pdf import export_report_pdf

START = datetime(2026, 9, 28, 14, 0)


def test_no_moments_when_everything_is_above_threshold() -> None:
    assert find_critical_moments([(0, 80.0), (5, 60.0)], 40) == []


def test_moment_ends_when_engagement_recovers() -> None:
    points = [(0, 80.0), (5, 30.0), (10, 20.0), (15, 70.0)]
    assert find_critical_moments(points, 40) == [CriticalMoment(5, 15, 20.0)]


def test_moment_open_at_the_end_of_lecture() -> None:
    points = [(0, 80.0), (5, 30.0), (10, 25.0)]
    assert find_critical_moments(points, 40) == [CriticalMoment(5, 10, 25.0)]


def test_several_moments() -> None:
    points = [(0, 30.0), (5, 60.0), (10, 10.0), (15, 90.0)]
    assert find_critical_moments(points, 40) == [
        CriticalMoment(0, 5, 30.0),
        CriticalMoment(10, 15, 10.0),
    ]


def _report(points: list[tuple[int, float]]) -> SessionReport:
    return SessionReport("id", START, 600, 40, points)


def test_summary_values() -> None:
    report = _report([(0, 80.0), (5, 20.0), (10, 50.0)])
    assert report.has_data
    assert report.average == 50.0
    assert report.minimum == 20.0
    assert len(report.critical_moments) == 1
    assert report.clock_time(90) == datetime(2026, 9, 28, 14, 1, 30)


def test_empty_report_has_no_data() -> None:
    report = _report([])
    assert not report.has_data
    assert report.average is None


def test_pdf_is_written(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    export_report_pdf(_report([(0, 80.0), (30, 25.0), (60, 70.0)]), path)
    assert path.read_bytes().startswith(b"%PDF")


def test_pdf_without_data_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export_report_pdf(_report([]), tmp_path / "report.pdf")


def test_short_moment_has_one_dip() -> None:
    points = [(0.0, 80.0), (5.0, 30.0), (10.0, 20.0), (15.0, 35.0), (20.0, 70.0)]
    assert find_dips(points, CriticalMoment(5.0, 20.0, 20.0)) == [10.0]


def test_long_moment_keeps_the_deepest_dips_apart() -> None:
    points = [(0.0, 30.0), (10.0, 10.0), (20.0, 12.0), (30.0, 11.0), (40.0, 30.0),
              (100.0, 15.0), (110.0, 30.0), (200.0, 25.0), (210.0, 35.0), (300.0, 5.0)]
    moment = CriticalMoment(0.0, 300.0, 5.0)
    assert find_dips(points, moment) == [10.0, 100.0, 300.0]


def test_mostly_low_lecture_is_detected() -> None:
    assert _report([(0, 30.0), (30, 25.0), (60, 70.0)]).mostly_below_threshold
    assert not _report([(0, 80.0), (30, 25.0), (60, 70.0), (90, 80.0)]).mostly_below_threshold


def test_transcript_is_split_into_fragments() -> None:
    assert MomentNote("первый\n\nвторой").transcript_parts == ["первый", "второй"]
    assert MomentNote().transcript_parts == []
