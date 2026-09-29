"""Tests for the video file signal check."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.video_source import FILE, FileUnreadableError, VideoSource, check_signal


def _write_video(path: Path, width: int = 64, height: int = 48, frames: int = 5) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
    for _ in range(frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


def test_valid_file_reports_its_parameters(tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    _write_video(path)
    info = check_signal(VideoSource(FILE, file_path=str(path)))
    assert (info.width, info.height) == (64, 48)
    assert info.frame.shape == (48, 64, 3)


def test_missing_file_is_unreadable(tmp_path: Path) -> None:
    with pytest.raises(FileUnreadableError):
        check_signal(VideoSource(FILE, file_path=str(tmp_path / "missing.mp4")))


def test_text_disguised_as_video_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "fake.mp4"
    path.write_text("not a video")
    with pytest.raises(FileUnreadableError):
        check_signal(VideoSource(FILE, file_path=str(path)))
