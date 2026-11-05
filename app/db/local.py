"""Local SQLite database: connection handling and initialization."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DB_PATH = Path.home() / ".focuscore" / "focuscore.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

LOCAL_USER_ID = "local"


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a connection that commits on success, rolls back on error and is
    always closed.

    The built-in `with sqlite3.connect(...)` only commits and never closes the
    connection, hence the wrapper.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the tables if missing and the default local user.

    Until accounts are implemented, every session belongs to LOCAL_USER_ID.
    Safe to call repeatedly.
    """
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection(db_path) as conn:
        conn.executescript(schema)
        _add_missing_columns(conn)
        conn.execute(
            "INSERT OR IGNORE INTO users (id, email, display_name, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                LOCAL_USER_ID,
                "local@focuscore",
                "Local user",
                datetime.now(timezone.utc).isoformat(),
            ),
        )


ADDED_COLUMNS = (("critical_moments", "transcript", "TEXT"),)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Add columns that appeared after the database was first created.

    CREATE TABLE IF NOT EXISTS does not change an existing table, so a
    database created by an earlier version gets the new columns here.
    """
    for table, column, column_type in ADDED_COLUMNS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
