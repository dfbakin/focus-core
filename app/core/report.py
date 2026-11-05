"""Lecture report: summary values and critical moments computed from the series."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.core.analysis import summarize

MIN_POINTS_FOR_REPORT = 1
MAX_DIPS_PER_MOMENT = 3
MIN_SECONDS_BETWEEN_DIPS = 60.0
MOSTLY_LOW_SHARE = 0.5
TRANSCRIPT_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class CriticalMoment:
    """A continuous stretch of the lecture with engagement below the threshold."""

    start_sec: float
    end_sec: float
    min_value: float


def find_critical_moments(
    points: list[tuple[float, float]], threshold: float
) -> list[CriticalMoment]:
    """Group consecutive points below the threshold into critical moments.

    A moment ends at the first point that is back at or above the threshold,
    or at the last low point if the lecture ended while engagement was low.
    """
    moments = []
    start: float | None = None
    lowest = 0.0
    last_low = 0.0
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


def find_dips(points: list[tuple[float, float]], moment: CriticalMoment) -> list[float]:
    """Return the offsets of the most pronounced drops inside a critical moment.

    A drop is a local minimum of the series. The deepest ones are taken
    first, at most MAX_DIPS_PER_MOMENT and at least MIN_SECONDS_BETWEEN_DIPS
    apart, so a short moment usually gives one drop and a long one several.
    The result is ordered by time.
    """
    inside = [(offset, value) for offset, value in points
              if moment.start_sec <= offset <= moment.end_sec]
    if not inside:
        return [moment.start_sec]
    minima = [
        inside[i] for i in range(len(inside))
        if (i == 0 or inside[i][1] <= inside[i - 1][1])
        and (i == len(inside) - 1 or inside[i][1] <= inside[i + 1][1])
    ]
    chosen: list[float] = []
    for offset, _ in sorted(minima, key=lambda point: point[1]):
        if all(abs(offset - other) >= MIN_SECONDS_BETWEEN_DIPS for other in chosen):
            chosen.append(offset)
        if len(chosen) == MAX_DIPS_PER_MOMENT:
            break
    return sorted(chosen)


@dataclass(frozen=True)
class MomentNote:
    """What is known about a critical moment besides the numbers.

    Attributes:
        transcript: Recognized speech of the teacher around the moment.
        comment: Text the teacher added later.
    """

    transcript: str | None = None
    comment: str | None = None

    @property
    def transcript_parts(self) -> list[str]:
        """Separate fragments of speech, one per drop of engagement."""
        if not self.transcript:
            return []
        return [part for part in self.transcript.split(TRANSCRIPT_SEPARATOR) if part]


def moment_key(start_sec: float) -> float:
    """Identify a moment by its start, rounded so that it survives storage."""
    return round(float(start_sec), 3)


@dataclass(frozen=True)
class SessionReport:
    """Everything needed to show or export the report of one lecture."""

    session_id: str
    started_at: datetime
    duration_sec: int | None
    threshold_pct: int
    points: list[tuple[float, float]]
    notes: dict[float, MomentNote] = field(default_factory=dict)

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

    @property
    def mostly_below_threshold(self) -> bool:
        """True if critical moments take more than half of the lecture.

        Then the threshold is probably too high for this audience.
        """
        if len(self.points) < 2:
            return False
        length = self.points[-1][0] - self.points[0][0]
        low = sum(moment.end_sec - moment.start_sec for moment in self.critical_moments)
        return length > 0 and low / length > MOSTLY_LOW_SHARE

    def note_for(self, moment: CriticalMoment) -> MomentNote:
        """Transcript and comment of a moment, empty if there are none."""
        return self.notes.get(moment_key(moment.start_sec), MomentNote())

    def clock_time(self, offset_sec: float) -> datetime:
        """Convert an offset from the lecture start into wall-clock time."""
        return self.started_at + timedelta(seconds=offset_sec)
