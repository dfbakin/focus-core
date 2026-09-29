CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    email       TEXT NOT NULL UNIQUE,
    display_name TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    duration_sec   INTEGER,
    avg_engagement REAL,
    min_engagement REAL,
    source_type    TEXT,
    threshold      INTEGER,
    synced_at      TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS engagement_points (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    offset_sec  INTEGER NOT NULL,
    value       REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_points_session ON engagement_points(session_id);

CREATE TABLE IF NOT EXISTS critical_moments (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    start_sec   INTEGER NOT NULL,
    end_sec     INTEGER NOT NULL,
    min_value   REAL,
    comment     TEXT,
    synced_at   TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

