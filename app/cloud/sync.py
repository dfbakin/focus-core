"""Two-way synchronization of finished sessions between SQLite and the cloud."""

from dataclasses import dataclass
from pathlib import Path

from app.cloud.client import CloudClient
from app.db.account import clear_pending_deletions, load_pending_deletions
from app.db.local import DB_PATH
from app.db.sessions import (
    insert_downloaded_session,
    mark_synced,
    point_rows,
    session_ids,
    unsynced_sessions,
)

SESSIONS_TABLE = "sessions"
POINTS_TABLE = "engagement_points"
POINTS_BATCH_SIZE = 500


@dataclass(frozen=True)
class SyncResult:
    """How many sessions went up to and came down from the server."""

    uploaded: int
    downloaded: int


def synchronize(client: CloudClient, user_id: str, db_path: Path = DB_PATH) -> SyncResult:
    """Apply pending deletions, upload new sessions, download missing ones.

    Deletions go first, so that a session deleted locally is not downloaded
    back from the server.
    """
    apply_pending_deletions(client, db_path)
    uploaded = upload_new_sessions(client, db_path)
    downloaded = download_missing_sessions(client, user_id, db_path)
    return SyncResult(uploaded=uploaded, downloaded=downloaded)


def apply_pending_deletions(client: CloudClient, db_path: Path = DB_PATH) -> None:
    """Delete on the server the sessions that were deleted locally."""
    pending = load_pending_deletions(db_path)
    if pending:
        client.delete(SESSIONS_TABLE, pending)
        clear_pending_deletions(db_path)


def upload_new_sessions(client: CloudClient, db_path: Path = DB_PATH) -> int:
    """Upload finished sessions that are not on the server yet, with their points."""
    sessions = unsynced_sessions(db_path)
    if not sessions:
        return 0
    client.upsert(SESSIONS_TABLE, sessions)
    for session in sessions:
        points = point_rows(session["id"], db_path)
        for start in range(0, len(points), POINTS_BATCH_SIZE):
            client.upsert(POINTS_TABLE, points[start:start + POINTS_BATCH_SIZE])
    mark_synced([session["id"] for session in sessions], db_path)
    return len(sessions)


def download_missing_sessions(
    client: CloudClient, user_id: str, db_path: Path = DB_PATH
) -> int:
    """Download server sessions that are absent locally, e.g. from another device."""
    local_ids = session_ids(db_path)
    remote = client.select(SESSIONS_TABLE, {"select": "*"})
    missing = [session for session in remote if session["id"] not in local_ids]
    for session in missing:
        points = client.select(
            POINTS_TABLE,
            {
                "select": "id,offset_sec,value",
                "session_id": f"eq.{session['id']}",
                "order": "offset_sec",
            },
        )
        insert_downloaded_session(session, points, user_id, db_path)
    return len(missing)
