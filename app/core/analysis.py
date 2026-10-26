"""Turning raw model scores into what the user sees.

Scores are smoothed with an exponential moving average (EMA) and mapped to
three engagement levels, so the indicator and the chart always agree.
"""

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


class ExponentialMovingAverage:
    """Exponential moving average with the smoothing set by a span.

    Each new value enters with weight alpha = 2 / (span + 1), and the weight of
    older values decays geometrically. The span plays the role of a window
    length: it roughly equals the number of recent estimates that shape the
    result. A span of 1 means no smoothing. Unlike a plain moving average, the
    curve reacts to a new value immediately and has no sharp jumps when an old
    value leaves the window.
    """

    def __init__(self, span: int) -> None:
        self._alpha = 2.0 / (span + 1)
        self._value: float | None = None

    def add(self, value: float) -> float:
        """Add a value and return the smoothed result."""
        if self._value is None:
            self._value = value
        else:
            self._value = self._alpha * value + (1 - self._alpha) * self._value
        return self._value


def summarize(values: list[float]) -> tuple[float, float] | None:
    """Return (average, minimum) of the values, or None if there are none."""
    if not values:
        return None
    return sum(values) / len(values), min(values)
