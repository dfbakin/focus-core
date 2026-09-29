"""Lecture report: summary values and critical moments computed from the series."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core.analysis import summarize

MIN_POINTS_FOR_REPORT = 1


@dataclass(frozen=True)
class CriticalMoment:
    """A continuous stretch of the lecture with engagement below the threshold."""

    start_sec: int
    end_sec: int
    min_value: float


def find_critical_moments(
    points: list[tuple[int, float]], threshold: float
) -> list[CriticalMoment]:
    """Group consecutive points below the threshold into critical moments.

    A moment ends at the first point that is back at or above the threshold,
    or at the last low point if the lecture ended while engagement was low.
    """
    moments = []
    start: int | None = None
    lowest = 0.0
    last_low = 0
    for offset, value in points:
        if value < threshold:
            if start is None:
                start, lowest = offset, value
            lowest = min(lowest, value)
            last_low = offset
        elif start is not None:
            moments.append(CriticalMoment(start, offset, lowest))
            start = None
    if start is not None:
        moments.append(CriticalMoment(start, last_low, lowest))
    return moments


@dataclass(frozen=True)
class SessionReport:
    """Everything needed to show or export the report of one lecture."""

    session_id: str
    started_at: datetime
    duration_sec: int | None
    threshold_pct: int
    points: list[tuple[int, float]]

    @property
    def has_data(self) -> bool:
        """False if no estimates were collected, e.g. the camera failed."""
        return len(self.points) >= MIN_POINTS_FOR_REPORT

    @property
    def average(self) -> float | None:
        """Mean engagement over the lecture."""
        summary = summarize([value for _, value in self.points])
        return summary[0] if summary else None

    @property
    def minimum(self) -> float | None:
        """Lowest engagement over the lecture."""
        summary = summarize([value for _, value in self.points])
        return summary[1] if summary else None

    @property
    def critical_moments(self) -> list[CriticalMoment]:
        """Stretches below the threshold that was set for this lecture."""
        return find_critical_moments(self.points, self.threshold_pct)

    def clock_time(self, offset_sec: int) -> datetime:
        """Convert an offset from the lecture start into wall-clock time."""
        return self.started_at + timedelta(seconds=offset_sec)
