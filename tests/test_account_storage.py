"""Tests for remembering the signed-in account on this device."""

from pathlib import Path

import pytest

from app.db.account import StoredAccount, load_account, save_account
from app.db.local import get_connection, init_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _account(user_id: str, email: str = "teacher@school.ru") -> StoredAccount:
    return StoredAccount(user_id, email, "Анна", "Иванова", "refresh")


def test_account_is_remembered(db_path: Path) -> None:
    save_account(_account("user-1"), db_path)
    assert load_account(db_path).user_id == "user-1"


def test_email_registered_again_gets_a_new_account(db_path: Path) -> None:
    save_account(_account("old-id"), db_path)
    save_account(_account("new-id"), db_path)
    assert load_account(db_path).user_id == "new-id"
    with get_connection(db_path) as conn:
        emails = {row["id"]: row["email"] for row in conn.execute("SELECT id, email FROM users")}
    assert emails["new-id"] == "teacher@school.ru"
    assert emails["old-id"] != "teacher@school.ru"
