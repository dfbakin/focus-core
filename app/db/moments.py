"""Transcripts and comments of critical moments (user scenario 6).

The moments themselves are computed from the engagement series every time
a report is built; this table stores only what cannot be recomputed. A
moment is identified by the session and its start offset.
"""

import uuid
from pathlib import Path

from app.core.report import CriticalMoment, MomentNote, moment_key
from app.db.local import DB_PATH, get_connection


def _upsert(conn, session_id: str, moment: CriticalMoment, column: str, value: str | None) -> None:
    """Set one column of a moment's row, creating the row if needed."""
    key = moment_key(moment.start_sec)
    updated = conn.execute(
        f"UPDATE critical_moments SET {column} = ? WHERE session_id = ? AND start_sec = ?",
        (value, session_id, key),
    ).rowcount
    if updated == 0:
        conn.execute(
            "INSERT INTO critical_moments "
            f"(id, session_id, start_sec, end_sec, min_value, {column}) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), session_id, key, moment.end_sec, moment.min_value, value),
        )


def save_transcripts(
    session_id: str,
    moments: list[CriticalMoment],
    texts: list[str | None],
    db_path: Path = DB_PATH,
) -> None:
    """Store the transcript of every moment; None leaves a moment without one."""
    with get_connection(db_path) as conn:
        for moment, text in zip(moments, texts):
            if text:
                _upsert(conn, session_id, moment, "transcript", text)


def save_comment(
    session_id: str, moment: CriticalMoment, comment: str, db_path: Path = DB_PATH
) -> None:
    """Store the teacher's comment; an empty text removes it."""
    with get_connection(db_path) as conn:
        _upsert(conn, session_id, moment, "comment", comment.strip() or None)


def load_moment_notes(session_id: str, db_path: Path = DB_PATH) -> dict[float, MomentNote]:
    """Return transcripts and comments of a session keyed by moment start."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT start_sec, transcript, comment FROM critical_moments WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    return {
        moment_key(row["start_sec"]): MomentNote(row["transcript"], row["comment"])
        for row in rows
    }
