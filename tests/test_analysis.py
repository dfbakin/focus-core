"""Tests for smoothing and engagement levels."""

import pytest

from app.core.analysis import EngagementLevel, MovingAverage, level_for_score, summarize


@pytest.mark.parametrize(
    ("score", "level"),
    [
        (0, EngagementLevel.LOW),
        (32.9, EngagementLevel.LOW),
        (33, EngagementLevel.MEDIUM),
        (65.9, EngagementLevel.MEDIUM),
        (66, EngagementLevel.HIGH),
        (100, EngagementLevel.HIGH),
    ],
)
def test_levels(score: float, level: EngagementLevel) -> None:
    assert level_for_score(score) == level


def test_moving_average_uses_only_last_values() -> None:
    average = MovingAverage(window=3)
    assert average.add(90) == 90
    assert average.add(10) == 50
    assert average.add(80) == 60
    assert average.add(0) == 30


def test_window_of_one_returns_raw_values() -> None:
    average = MovingAverage(window=1)
    average.add(90)
    assert average.add(10) == 10


def test_summary() -> None:
    assert summarize([]) is None
    assert summarize([20.0, 60.0, 40.0]) == (40.0, 20.0)
