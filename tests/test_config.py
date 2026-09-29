"""Tests for the analysis settings validation rules."""

import pytest

from app.core.config import is_interval_too_short, is_threshold_degenerate


@pytest.mark.parametrize("value", [0, 100])
def test_threshold_extremes_are_degenerate(value: int) -> None:
    assert is_threshold_degenerate(value)


@pytest.mark.parametrize("value", [1, 40, 99])
def test_regular_threshold_is_accepted(value: int) -> None:
    assert not is_threshold_degenerate(value)


@pytest.mark.parametrize("value", [0.1, 0.5, 0.9])
def test_subsecond_interval_is_too_short(value: float) -> None:
    assert is_interval_too_short(value)


@pytest.mark.parametrize("value", [1.0, 5.0, 60.0])
def test_interval_from_one_second_is_accepted(value: float) -> None:
    assert not is_interval_too_short(value)
