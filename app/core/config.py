"""Analysis settings and their validation rules.

Pure Python module with no Qt or database dependencies, so the rules can be
unit-tested in isolation.
"""

from dataclasses import dataclass

THRESHOLD_MIN = 0
THRESHOLD_MAX = 100
INTERVAL_MIN_SEC = 0.1
INTERVAL_MAX_SEC = 60.0
INTERVAL_SAFE_MIN_SEC = 1.0


@dataclass(frozen=True)
class AnalysisSettings:
    """Parameters applied to the next analysis session.

    Attributes:
        threshold_pct: Engagement level (0-100) below which a moment is critical.
        interval_sec: Time between two consecutive model estimates, in seconds.
        smoothing_window: Number of consecutive estimates averaged together.
    """

    threshold_pct: int = 40
    interval_sec: float = 5.0
    smoothing_window: int = 3


def is_threshold_degenerate(threshold_pct: int) -> bool:
    """Return True if no critical moments can be detected with this threshold."""
    return threshold_pct in (THRESHOLD_MIN, THRESHOLD_MAX)


def is_interval_too_short(interval_sec: float) -> bool:
    """Return True if the model cannot keep up with this estimation interval."""
    return interval_sec < INTERVAL_SAFE_MIN_SEC
