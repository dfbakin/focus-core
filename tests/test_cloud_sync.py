"""Tests for synchronization and account logic on a fake cloud."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.cloud.account import AccountService
from app.cloud.client import AuthSession, CloudUnavailableError, InvalidCredentialsError
from app.core.video_source import CAMERA
from app.db.account import load_account, load_pending_deletions
from app.db.local import LOCAL_USER_ID, get_connection, init_db
from app.db.sessions import (
    create_session,
    delete_sessions,
    finish_session,
    list_sessions,
    load_engagement_points,
    save_engagement_points,
)

START = datetime(2026, 9, 28, 10, 0)


class FakeCloud:
    """In-memory stand-in for CloudClient with one registered user."""

    def __init__(self) -> None:
        self.tables: dict[str, dict[str, dict]] = {"sessions": {}, "engagement_points": {}}
        self.session: AuthSession | None = None
        self.password = "secret1"
        self.names = ("Анна", "Иванова")

    def _login(self) -> AuthSession:
        self.session = AuthSession("user-1", "a@b.ru", *self.names, "access", "refresh")
        return self.session

    def sign_up(self, email, password, first_name, last_name) -> AuthSession:
        self.names = (first_name, last_name)
        return self._login()

    def sign_in(self, email, password) -> AuthSession:
        if password != self.password:
            raise InvalidCredentialsError("bad")
        return self._login()

    def restore(self, refresh_token) -> AuthSession:
        return self._login()

    def update_profile(self, first_name, last_name) -> AuthSession:
        self.names = (first_name, last_name)
        return self._login()

    def change_password(self, new_password) -> None:
        self.password = new_password

    def select(self, table, params) -> list[dict]:
        rows = list(self.tables[table].values())
        if "session_id" in params:
            wanted = params["session_id"].removeprefix("eq.")
            rows = [row for row in rows if row["session_id"] == wanted]
        return rows

    def upsert(self, table, rows) -> None:
        for row in rows:
            self.tables[table][row["id"]] = dict(row)

    def delete(self, table, ids) -> None:
        for session_id in ids:
            self.tables[table].pop(session_id, None)
            if table == "sessions":
                points = self.tables["engagement_points"]
                for key in [k for k, p in points.items() if p["session_id"] == session_id]:
                    del points[key]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _finished(db_path: Path, minutes_ago: int = 60) -> str:
    started = START - timedelta(minutes=minutes_ago)
    session_id = create_session(CAMERA, 40, db_path, started_at=started)
    finish_session(session_id, db_path, finished_at=started + timedelta(minutes=45))
    save_engagement_points(session_id, [(0, 70.0), (30, 20.0)], db_path)
    return session_id


def test_sign_in_uploads_local_sessions_and_adopts_them(db_path: Path) -> None:
    cloud = FakeCloud()
    session_id = _finished(db_path)
    result = AccountService(cloud, db_path).sign_in("a@b.ru", "secret1")

    assert result.uploaded == 1
    assert session_id in cloud.tables["sessions"]
    assert len(cloud.tables["engagement_points"]) == 2
    with get_connection(db_path) as conn:
        owner = conn.execute("SELECT user_id FROM sessions").fetchone()[0]
    assert owner == "user-1"
    assert load_account(db_path).display_name == "Анна Иванова"


def test_sessions_from_another_device_are_downloaded(db_path: Path) -> None:
    cloud = FakeCloud()
    cloud.upsert("sessions", [{
        "id": "remote-1", "started_at": START.isoformat(),
        "finished_at": (START + timedelta(minutes=30)).isoformat(),
        "duration_sec": 1800, "avg_engagement": 55.0, "min_engagement": 30.0,
        "source_type": CAMERA, "threshold": 40,
    }])
    cloud.upsert("engagement_points", [
        {"id": "p1", "session_id": "remote-1", "offset_sec": 0, "value": 55.0},
    ])
    result = AccountService(cloud, db_path).sign_in("a@b.ru", "secret1")

    assert result.downloaded == 1
    assert [s.id for s in list_sessions(db_path)] == ["remote-1"]
    assert load_engagement_points("remote-1", db_path) == [(0, 55.0)]


def test_second_sync_uploads_nothing_new(db_path: Path) -> None:
    cloud = FakeCloud()
    _finished(db_path)
    service = AccountService(cloud, db_path)
    service.sign_in("a@b.ru", "secret1")
    assert service.synchronize().uploaded == 0


def test_local_deletion_reaches_the_server(db_path: Path) -> None:
    cloud = FakeCloud()
    session_id = _finished(db_path)
    service = AccountService(cloud, db_path)
    service.sign_in("a@b.ru", "secret1")

    delete_sessions([session_id], db_path)
    assert load_pending_deletions(db_path) == [session_id]
    result = service.synchronize()

    assert session_id not in cloud.tables["sessions"]
    assert result.downloaded == 0
    assert load_pending_deletions(db_path) == []


def test_deletions_without_account_are_not_recorded(db_path: Path) -> None:
    delete_sessions([_finished(db_path)], db_path)
    assert load_pending_deletions(db_path) == []


def test_wrong_password_signs_nobody_in(db_path: Path) -> None:
    with pytest.raises(InvalidCredentialsError):
        AccountService(FakeCloud(), db_path).sign_in("a@b.ru", "wrong")
    assert load_account(db_path) is None


def test_offline_without_configuration(db_path: Path) -> None:
    with pytest.raises(CloudUnavailableError):
        AccountService(None, db_path).sign_in("a@b.ru", "secret1")


def test_profile_and_password_changes(db_path: Path) -> None:
    cloud = FakeCloud()
    service = AccountService(cloud, db_path)
    service.sign_in("a@b.ru", "secret1")

    service.update_profile("Мария", "Петрова")
    assert load_account(db_path).display_name == "Мария Петрова"

    with pytest.raises(InvalidCredentialsError):
        service.change_password("wrong", "newpass1")
    service.change_password("secret1", "newpass1")
    assert cloud.password == "newpass1"


def test_sign_out_hides_account_sessions_until_next_sign_in(db_path: Path) -> None:
    service = AccountService(FakeCloud(), db_path)
    _finished(db_path)
    service.sign_in("a@b.ru", "secret1")
    service.sign_out()
    assert load_account(db_path) is None
    assert list_sessions(db_path) == []

    service.sign_in("a@b.ru", "secret1")
    assert len(list_sessions(db_path)) == 1


def test_sessions_recorded_after_sign_out_are_separate(db_path: Path) -> None:
    service = AccountService(FakeCloud(), db_path)
    service.sign_in("a@b.ru", "secret1")
    _finished(db_path)
    service.sign_out()
    local = _finished(db_path, minutes_ago=10)
    assert [s.id for s in list_sessions(db_path)] == [local]


def test_new_sessions_belong_to_signed_in_user(db_path: Path) -> None:
    AccountService(FakeCloud(), db_path).sign_in("a@b.ru", "secret1")
    create_session(CAMERA, 40, db_path)
    with get_connection(db_path) as conn:
        owners = {row[0] for row in conn.execute("SELECT user_id FROM sessions")}
    assert owners == {"user-1"} and LOCAL_USER_ID not in owners
