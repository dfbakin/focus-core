"""Data retention policy: how long finished sessions are kept."""

from dataclasses import dataclass
from datetime import datetime, timedelta

RETENTION_CHOICES_DAYS = [7, 14, 30, 90, 180, 365]
DEFAULT_RETENTION_DAYS = 30


@dataclass(frozen=True)
class RetentionPolicy:
    """Whether old sessions are removed automatically and after how many days."""

    auto_delete: bool = False
    days: int = DEFAULT_RETENTION_DAYS


def expiry_cutoff(days: int, now: datetime | None = None) -> datetime:
    """Return the moment before which sessions are considered expired."""
    return (now or datetime.now()) - timedelta(days=days)
