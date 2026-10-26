"""Tests for choosing clips from a video file."""

import pytest

from app.core.offline import (
    DEFAULT_FPS,
    forecast_processing_time,
    next_countdown_value,
    frames_between_estimates,
    iter_clips,
    usable_fps,
)


@pytest.mark.parametrize("reported", [0.0, -1.0, float("nan")])
def test_unknown_frame_rate_falls_back_to_default(reported: float) -> None:
    assert usable_fps(reported) == DEFAULT_FPS


def test_known_frame_rate_is_kept() -> None:
    assert usable_fps(30.0) == 30.0


@pytest.mark.parametrize(
    ("interval", "fps", "expected"),
    [(1.0, 30.0, 30), (0.2, 30.0, 6), (0.2, 25.0, 5), (0.01, 30.0, 1)],
)
def test_frames_between_estimates(interval: float, fps: float, expected: int) -> None:
    assert frames_between_estimates(interval, fps) == expected


def test_no_clip_until_window_is_filled() -> None:
    assert list(iter_clips(range(15), fps=10.0, interval_sec=0.1, num_frames=4, stride=4)) == []


def test_first_clip_uses_every_stride_frame_of_the_window() -> None:
    offset, clip = next(iter_clips(range(100), fps=10.0, interval_sec=1.0, num_frames=4, stride=4))
    assert clip == [0, 4, 8, 12]
    assert offset == pytest.approx(1.6)


def test_clips_are_spaced_by_video_time() -> None:
    clips = list(iter_clips(range(46), fps=10.0, interval_sec=1.0, num_frames=4, stride=4))
    assert [offset for offset, _ in clips] == pytest.approx([1.6, 2.6, 3.6, 4.6])
    assert clips[1][1] == [10, 14, 18, 22]


def test_no_forecast_at_the_very_beginning() -> None:
    assert forecast_processing_time(elapsed_sec=0.5, fraction_done=0.3) is None
    assert forecast_processing_time(elapsed_sec=10.0, fraction_done=0.01) is None


def test_forecast_assumes_constant_speed() -> None:
    total, remaining = forecast_processing_time(elapsed_sec=20.0, fraction_done=0.25)
    assert total == pytest.approx(80.0)
    assert remaining == pytest.approx(60.0)


def test_nothing_remains_when_done() -> None:
    _, remaining = forecast_processing_time(elapsed_sec=30.0, fraction_done=1.0)
    assert remaining == 0.0


def test_countdown_starts_from_the_first_forecast() -> None:
    assert next_countdown_value(None, 40.0) == 40.0


def test_countdown_follows_a_lower_forecast() -> None:
    assert next_countdown_value(30.0, 28.5) == 28.5


def test_countdown_ignores_a_small_jump_up() -> None:
    assert next_countdown_value(30.0, 32.0) == 30.0


def test_countdown_accepts_a_real_slowdown() -> None:
    assert next_countdown_value(30.0, 40.0) == 40.0

