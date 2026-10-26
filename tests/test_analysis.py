"""Tests for smoothing and engagement levels."""

import pytest

from app.core.analysis import (
    EngagementLevel,
    ExponentialMovingAverage,
    level_for_score,
    summarize,
)


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


def test_first_value_is_returned_as_is() -> None:
    assert ExponentialMovingAverage(span=3).add(80) == 80


def test_span_three_gives_half_weight_to_new_value() -> None:
    average = ExponentialMovingAverage(span=3)
    average.add(80)
    assert average.add(20) == pytest.approx(50)
    assert average.add(20) == pytest.approx(35)


def test_span_one_returns_raw_values() -> None:
    average = ExponentialMovingAverage(span=1)
    average.add(90)
    assert average.add(10) == 10


def test_larger_span_is_smoother() -> None:
    small, large = ExponentialMovingAverage(2), ExponentialMovingAverage(8)
    for value in (90, 10):
        smooth_small, smooth_large = small.add(value), large.add(value)
    assert smooth_large > smooth_small


def test_summary() -> None:
    assert summarize([]) is None
    assert summarize([20.0, 60.0, 40.0]) == (40.0, 20.0)
