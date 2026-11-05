"""Tests for analysing a whole video file without the interface and the model."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.cli import report_to_dict
from app.core.file_analysis import (
    VideoUnreadableError,
    analyze_video_file,
    build_report,
)


def _write_video(path: Path, seconds: int, fps: int = 10) -> None:
    """Write a video whose brightness grows with time."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for index in range(seconds * fps):
        writer.write(np.full((48, 64, 3), index % 250, np.uint8))
    writer.release()


def _brightness_score(clip: list[np.ndarray]) -> float:
    """Fake model: the brighter the last frame, the lower the engagement."""
    return 100.0 - float(clip[-1].mean()) / 2.5


def test_estimates_follow_video_time(tmp_path: Path) -> None:
    path = tmp_path / "lecture.mp4"
    _write_video(path, seconds=20)
    analysis = analyze_video_file(path, _brightness_score, interval_sec=2.0, smoothing_window=1)
    offsets = [offset for offset, _ in analysis.points]
    assert offsets[0] == pytest.approx(6.4)
    assert offsets[1] - offsets[0] == pytest.approx(2.0)
    assert analysis.duration_sec == pytest.approx(20.0)
    assert analysis.points[-1][1] < analysis.points[0][1]


def test_report_lists_critical_moments(tmp_path: Path) -> None:
    path = tmp_path / "lecture.mp4"
    _write_video(path, seconds=20)
    analysis = analyze_video_file(path, _brightness_score, interval_sec=1.0, smoothing_window=1)
    report = build_report(analysis, threshold_pct=60)
    data = report_to_dict(report, path)
    assert data["video"] == "lecture.mp4"
    assert len(data["points"]) == len(analysis.points)
    assert data["critical_moments"]
    json.dumps(data)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VideoUnreadableError):
        analyze_video_file(tmp_path / "missing.mp4", _brightness_score, 1.0, 1)
