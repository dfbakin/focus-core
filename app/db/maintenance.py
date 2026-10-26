"""Storage maintenance: removing expired sessions and measuring the database."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.core.retention import RetentionPolicy, expiry_cutoff
from app.db.account import current_user_id, record_remote_deletions
from app.db.local import DB_PATH, get_connection


@dataclass(frozen=True)
class StorageStatus:
    """What is stored at the moment."""

    session_count: int
    size_bytes: int


@dataclass(frozen=True)
class CleanupResult:
    """Outcome of removing expired sessions."""

    deleted_count: int
    freed_bytes: int


def storage_status(db_path: Path = DB_PATH) -> StorageStatus:
    """Count finished sessions of the current user and measure the database file."""
    with get_connection(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE finished_at IS NOT NULL AND user_id = ?",
            (current_user_id(db_path),),
        ).fetchone()[0]
    return StorageStatus(session_count=count, size_bytes=db_path.stat().st_size)


def delete_expired_sessions(
    days: int, db_path: Path = DB_PATH, now: datetime | None = None
) -> CleanupResult:
    """Delete the current user's sessions older than `days` and compact the file.

    SQLite does not shrink its file after DELETE; VACUUM rewrites it without
    the free pages, so the freed space can actually be measured.
    """
    cutoff = expiry_cutoff(days, now).isoformat()
    user_id = current_user_id(db_path)
    size_before = db_path.stat().st_size
    with get_connection(db_path) as conn:
        expired = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM sessions WHERE started_at < ? AND user_id = ?",
                (cutoff, user_id),
            ).fetchall()
        ]
        conn.execute(
            "DELETE FROM sessions WHERE started_at < ? AND user_id = ?", (cutoff, user_id)
        )
    deleted = len(expired)
    record_remote_deletions(expired, db_path)
    if deleted:
        with get_connection(db_path) as conn:
            conn.execute("VACUUM")
    freed = max(0, size_before - db_path.stat().st_size)
    return CleanupResult(deleted_count=deleted, freed_bytes=freed)


def apply_retention_policy(
    policy: RetentionPolicy, db_path: Path = DB_PATH
) -> CleanupResult | None:
    """Delete expired sessions if automatic deletion is on; called at startup."""
    if not policy.auto_delete:
        return None
    return delete_expired_sessions(policy.days, db_path)
