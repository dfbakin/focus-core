"""The signed-in cloud account and pending remote deletions, stored locally."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.db.local import DB_PATH, LOCAL_USER_ID, get_connection
from app.db.settings import delete_settings, read_all_settings, write_settings

USER_ID_KEY = "account_user_id"
EMAIL_KEY = "account_email"
FIRST_NAME_KEY = "account_first_name"
LAST_NAME_KEY = "account_last_name"
REFRESH_TOKEN_KEY = "account_refresh_token"
PENDING_DELETIONS_KEY = "pending_remote_deletions"
ACCOUNT_KEYS = [USER_ID_KEY, EMAIL_KEY, FIRST_NAME_KEY, LAST_NAME_KEY, REFRESH_TOKEN_KEY]


@dataclass(frozen=True)
class StoredAccount:
    """Cloud account remembered on this device between launches."""

    user_id: str
    email: str
    first_name: str
    last_name: str
    refresh_token: str

    @property
    def display_name(self) -> str:
        """First and last name, or the email if the name is empty."""
        name = f"{self.first_name} {self.last_name}".strip()
        return name or self.email


def load_account(db_path: Path = DB_PATH) -> StoredAccount | None:
    """Return the remembered account, or None if nobody is signed in."""
    stored = read_all_settings(db_path)
    if USER_ID_KEY not in stored:
        return None
    return StoredAccount(
        user_id=stored[USER_ID_KEY],
        email=stored.get(EMAIL_KEY, ""),
        first_name=stored.get(FIRST_NAME_KEY, ""),
        last_name=stored.get(LAST_NAME_KEY, ""),
        refresh_token=stored.get(REFRESH_TOKEN_KEY, ""),
    )


def save_account(account: StoredAccount, db_path: Path = DB_PATH) -> None:
    """Remember the account and make sure it exists in the users table."""
    write_settings(
        {
            USER_ID_KEY: account.user_id,
            EMAIL_KEY: account.email,
            FIRST_NAME_KEY: account.first_name,
            LAST_NAME_KEY: account.last_name,
            REFRESH_TOKEN_KEY: account.refresh_token,
        },
        db_path,
    )
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO users (id, email, display_name, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET email = excluded.email, "
            "display_name = excluded.display_name",
            (
                account.user_id,
                account.email,
                account.display_name,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def clear_account(db_path: Path = DB_PATH) -> None:
    """Forget the account on this device; local sessions are kept."""
    delete_settings(ACCOUNT_KEYS + [PENDING_DELETIONS_KEY], db_path)


def current_user_id(db_path: Path = DB_PATH) -> str:
    """Owner of new sessions: the signed-in account or the local user."""
    account = load_account(db_path)
    return account.user_id if account else LOCAL_USER_ID


def load_pending_deletions(db_path: Path = DB_PATH) -> list[str]:
    """Ids of sessions deleted locally but not yet on the server."""
    raw = read_all_settings(db_path).get(PENDING_DELETIONS_KEY)
    return json.loads(raw) if raw else []


def record_remote_deletions(session_ids: list[str], db_path: Path = DB_PATH) -> None:
    """Remember deleted ids so the next synchronization removes them remotely.

    Nothing is recorded without an account: then there is no server copy.
    """
    if not session_ids or load_account(db_path) is None:
        return
    pending = load_pending_deletions(db_path)
    pending.extend(i for i in session_ids if i not in pending)
    write_settings({PENDING_DELETIONS_KEY: json.dumps(pending)}, db_path)


def clear_pending_deletions(db_path: Path = DB_PATH) -> None:
    """Forget pending deletions after they were applied on the server."""
    delete_settings([PENDING_DELETIONS_KEY], db_path)
