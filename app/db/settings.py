"""Persistence of user settings in the key-value `settings` table."""

from dataclasses import asdict, fields
from pathlib import Path

from app.core.config import AnalysisSettings
from app.core.diagnostics import DEFAULT_MIN_FREE_RAM_MB
from app.core.retention import DEFAULT_RETENTION_DAYS, RetentionPolicy
from app.core.video_source import CAMERA, FILE, VideoSource
from app.db.local import DB_PATH, get_connection

SOURCE_KIND_KEY = "source_kind"
SOURCE_CAMERA_KEY = "source_camera_index"
SOURCE_CAMERA_NAME_KEY = "source_camera_name"
SOURCE_FILE_KEY = "source_file_path"
MIN_FREE_RAM_KEY = "min_free_ram_mb"
SIMULATED_ERROR_RATE_KEY = "simulated_error_rate"
AUTO_DELETE_KEY = "retention_auto_delete"
RETENTION_DAYS_KEY = "retention_days"


def read_all_settings(db_path: Path) -> dict[str, str]:
    """Return every stored setting as a {key: value} dictionary."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {row["key"]: row["value"] for row in rows}


def delete_settings(keys: list[str], db_path: Path) -> None:
    """Remove the given keys."""
    with get_connection(db_path) as conn:
        conn.executemany("DELETE FROM settings WHERE key = ?", [(key,) for key in keys])


def write_settings(values: dict[str, object], db_path: Path) -> None:
    """Insert new keys and overwrite existing ones in a single transaction."""
    rows = [(key, str(value)) for key, value in values.items()]
    with get_connection(db_path) as conn:
        conn.executemany(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            rows,
        )


def load_settings(db_path: Path = DB_PATH) -> AnalysisSettings:
    """Read analysis settings.

    Missing or malformed values fall back to the defaults of AnalysisSettings.
    """
    stored = read_all_settings(db_path)
    values = {}
    for field in fields(AnalysisSettings):
        raw = stored.get(field.name)
        if raw is None:
            continue
        try:
            values[field.name] = field.type(raw)
        except ValueError:
            continue
    return AnalysisSettings(**values)


def save_settings(settings: AnalysisSettings, db_path: Path = DB_PATH) -> None:
    """Store every field of the analysis settings."""
    write_settings(asdict(settings), db_path)


def load_video_source(db_path: Path = DB_PATH) -> VideoSource | None:
    """Return the saved video source, or None if it was never chosen."""
    stored = read_all_settings(db_path)
    kind = stored.get(SOURCE_KIND_KEY)
    if kind == CAMERA:
        try:
            return VideoSource(
                CAMERA,
                camera_index=int(stored[SOURCE_CAMERA_KEY]),
                camera_name=stored.get(SOURCE_CAMERA_NAME_KEY, ""),
            )
        except (KeyError, ValueError):
            return None
    if kind == FILE:
        return VideoSource(FILE, file_path=stored.get(SOURCE_FILE_KEY, ""))
    return None


def save_video_source(source: VideoSource, db_path: Path = DB_PATH) -> None:
    """Store the chosen video source."""
    write_settings(
        {
            SOURCE_KIND_KEY: source.kind,
            SOURCE_CAMERA_KEY: source.camera_index,
            SOURCE_CAMERA_NAME_KEY: source.camera_name,
            SOURCE_FILE_KEY: source.file_path,
        },
        db_path,
    )


def load_min_free_ram_mb(db_path: Path = DB_PATH) -> int:
    """Return the free RAM required to start analysis, in megabytes."""
    try:
        return int(read_all_settings(db_path)[MIN_FREE_RAM_KEY])
    except (KeyError, ValueError):
        return DEFAULT_MIN_FREE_RAM_MB


def save_min_free_ram_mb(megabytes: int, db_path: Path = DB_PATH) -> None:
    """Store the free RAM required to start analysis."""
    write_settings({MIN_FREE_RAM_KEY: megabytes}, db_path)


def load_simulated_error_rate(db_path: Path = DB_PATH) -> float:
    """Return the share of estimates deliberately failed for testing (0 by default)."""
    try:
        return float(read_all_settings(db_path)[SIMULATED_ERROR_RATE_KEY])
    except (KeyError, ValueError):
        return 0.0


def save_simulated_error_rate(rate: float, db_path: Path = DB_PATH) -> None:
    """Store the share of estimates to fail on purpose, from 0 to 1."""
    write_settings({SIMULATED_ERROR_RATE_KEY: rate}, db_path)


def load_retention_policy(db_path: Path = DB_PATH) -> RetentionPolicy:
    """Return the stored retention policy or the default one."""
    stored = read_all_settings(db_path)
    try:
        days = int(stored.get(RETENTION_DAYS_KEY, DEFAULT_RETENTION_DAYS))
    except ValueError:
        days = DEFAULT_RETENTION_DAYS
    return RetentionPolicy(auto_delete=stored.get(AUTO_DELETE_KEY) == "1", days=days)


def save_retention_policy(policy: RetentionPolicy, db_path: Path = DB_PATH) -> None:
    """Store the retention policy."""
    write_settings(
        {AUTO_DELETE_KEY: int(policy.auto_delete), RETENTION_DAYS_KEY: policy.days},
        db_path,
    )
