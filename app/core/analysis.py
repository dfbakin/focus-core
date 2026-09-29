"""Turning raw model scores into what the user sees.

Scores are smoothed with a moving average and mapped to three engagement
levels, so the indicator and the chart always agree with each other.
"""

from collections import deque
from enum import Enum

MEDIUM_MIN_SCORE = 33.0
HIGH_MIN_SCORE = 66.0


class EngagementLevel(Enum):
    """Engagement class shown by the colored indicator."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def level_for_score(score: float) -> EngagementLevel:
    """Map a score 0..100 to an engagement level."""
    if score >= HIGH_MIN_SCORE:
        return EngagementLevel.HIGH
    if score >= MEDIUM_MIN_SCORE:
        return EngagementLevel.MEDIUM
    return EngagementLevel.LOW


class MovingAverage:
    """Average of the last `window` values."""

    def __init__(self, window: int) -> None:
        self._values: deque[float] = deque(maxlen=window)

    def add(self, value: float) -> float:
        """Add a value and return the average of the current window."""
        self._values.append(value)
        return sum(self._values) / len(self._values)


def summarize(values: list[float]) -> tuple[float, float] | None:
    """Return (average, minimum) of the values, or None if there are none."""
    if not values:
        return None
    return sum(values) / len(values), min(values)
