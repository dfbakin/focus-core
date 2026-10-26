"""Storage of lecture sessions in the `sessions` table."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.core.report import SessionReport
from app.db.account import current_user_id, record_remote_deletions
from app.db.local import DB_PATH, get_connection


@dataclass(frozen=True)
class SessionSummary:
    """One row of the session history."""

    id: str
    started_at: datetime
    duration_sec: int | None
    avg_engagement: float | None


def _now() -> datetime:
    """Current local time without microseconds."""
    return datetime.now().replace(microsecond=0)


def create_session(
    source_kind: str,
    threshold_pct: int,
    db_path: Path = DB_PATH,
    started_at: datetime | None = None,
) -> str:
    """Insert a new unfinished session owned by the current user; return its id."""
    session_id = str(uuid.uuid4())
    started = started_at or _now()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (id, user_id, started_at, source_type, threshold) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                current_user_id(db_path),
                started.isoformat(),
                source_kind,
                threshold_pct,
            ),
        )
    return session_id


def finish_session(
    session_id: str,
    db_path: Path = DB_PATH,
    finished_at: datetime | None = None,
    duration_sec: int | None = None,
) -> int:
    """Mark the session as finished and return its duration in seconds.

    `duration_sec` is the active time without pauses. If it is not given,
    the duration is the wall-clock time between start and finish.
    """
    finished = finished_at or _now()
    with get_connection(db_path) as conn:
        if duration_sec is None:
            row = conn.execute(
                "SELECT started_at FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            started = datetime.fromisoformat(row["started_at"])
            duration_sec = int((finished - started).total_seconds())
        duration = duration_sec
        conn.execute(
            "UPDATE sessions SET finished_at = ?, duration_sec = ? WHERE id = ?",
            (finished.isoformat(), duration, session_id),
        )
    return duration


def save_engagement_summary(
    session_id: str,
    avg_engagement: float,
    min_engagement: float,
    db_path: Path = DB_PATH,
) -> None:
    """Store aggregated engagement values of a session."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE sessions SET avg_engagement = ?, min_engagement = ? WHERE id = ?",
            (avg_engagement, min_engagement, session_id),
        )


def save_engagement_points(
    session_id: str,
    points: list[tuple[float, float]],
    db_path: Path = DB_PATH,
) -> None:
    """Store the engagement time series as (offset in seconds, value) pairs."""
    rows = [(str(uuid.uuid4()), session_id, offset, value) for offset, value in points]
    with get_connection(db_path) as conn:
        conn.executemany(
            "INSERT INTO engagement_points (id, session_id, offset_sec, value) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )


def load_engagement_points(
    session_id: str, db_path: Path = DB_PATH
) -> list[tuple[float, float]]:
    """Return the engagement time series of a session ordered by time."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT offset_sec, value FROM engagement_points "
            "WHERE session_id = ? ORDER BY offset_sec",
            (session_id,),
        ).fetchall()
    return [(row["offset_sec"], row["value"]) for row in rows]


def list_sessions(db_path: Path = DB_PATH) -> list[SessionSummary]:
    """Return finished sessions of the current user, the most recent first.

    Sessions of an account that signed out stay on the device but are hidden
    until that account signs in again.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, started_at, duration_sec, avg_engagement FROM sessions "
            "WHERE finished_at IS NOT NULL AND user_id = ? ORDER BY started_at DESC",
            (current_user_id(db_path),),
        ).fetchall()
    return [
        SessionSummary(
            id=row["id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            duration_sec=row["duration_sec"],
            avg_engagement=row["avg_engagement"],
        )
        for row in rows
    ]


def delete_sessions(session_ids: list[str], db_path: Path = DB_PATH) -> None:
    """Delete sessions together with their points and critical moments.

    Related rows are removed by ON DELETE CASCADE in the schema.
    """
    if not session_ids:
        return
    placeholders = ", ".join("?" for _ in session_ids)
    with get_connection(db_path) as conn:
        conn.execute(f"DELETE FROM sessions WHERE id IN ({placeholders})", session_ids)
    record_remote_deletions(session_ids, db_path)


def load_session_report(session_id: str, db_path: Path = DB_PATH) -> SessionReport | None:
    """Collect the session record and its series into a report, or None."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT id, started_at, duration_sec, threshold FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return SessionReport(
        session_id=row["id"],
        started_at=datetime.fromisoformat(row["started_at"]),
        duration_sec=row["duration_sec"],
        threshold_pct=row["threshold"] if row["threshold"] is not None else 0,
        points=load_engagement_points(session_id, db_path),
    )


SYNCED_SESSION_FIELDS = (
    "id",
    "started_at",
    "finished_at",
    "duration_sec",
    "avg_engagement",
    "min_engagement",
    "source_type",
    "threshold",
)


def unsynced_sessions(db_path: Path = DB_PATH) -> list[dict]:
    """Finished sessions of the current user not uploaded yet, as dictionaries."""
    columns = ", ".join(SYNCED_SESSION_FIELDS)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT {columns} FROM sessions "
            "WHERE finished_at IS NOT NULL AND synced_at IS NULL AND user_id = ?",
            (current_user_id(db_path),),
        ).fetchall()
    return [dict(row) for row in rows]


def point_rows(session_id: str, db_path: Path = DB_PATH) -> list[dict]:
    """Engagement points of a session with their ids, for uploading."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, session_id, offset_sec, value FROM engagement_points "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def mark_synced(session_ids: list[str], db_path: Path = DB_PATH) -> None:
    """Remember that these sessions are stored on the server."""
    synced_at = _now().isoformat()
    with get_connection(db_path) as conn:
        conn.executemany(
            "UPDATE sessions SET synced_at = ? WHERE id = ?",
            [(synced_at, session_id) for session_id in session_ids],
        )


def session_ids(db_path: Path = DB_PATH) -> set[str]:
    """Ids of all sessions stored locally."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT id FROM sessions").fetchall()
    return {row["id"] for row in rows}


def insert_downloaded_session(
    session: dict, points: list[dict], user_id: str, db_path: Path = DB_PATH
) -> None:
    """Store a session received from the server together with its points."""
    columns = ", ".join(SYNCED_SESSION_FIELDS)
    placeholders = ", ".join("?" for _ in SYNCED_SESSION_FIELDS)
    values = [session.get(field) for field in SYNCED_SESSION_FIELDS]
    with get_connection(db_path) as conn:
        conn.execute(
            f"INSERT INTO sessions ({columns}, user_id, synced_at) "
            f"VALUES ({placeholders}, ?, ?)",
            [*values, user_id, _now().isoformat()],
        )
        conn.executemany(
            "INSERT INTO engagement_points (id, session_id, offset_sec, value) "
            "VALUES (?, ?, ?, ?)",
            [(p["id"], session["id"], p["offset_sec"], p["value"]) for p in points],
        )


def reassign_sessions(from_user: str, to_user: str, db_path: Path = DB_PATH) -> None:
    """Give sessions recorded before signing in to the account that signed in."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE sessions SET user_id = ? WHERE user_id = ?", (to_user, from_user)
        )
