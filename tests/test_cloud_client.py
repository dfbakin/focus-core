"""Tests for the Supabase client on a fake HTTP transport."""

import pytest
import requests

from app.cloud.client import (
    CloudClient,
    CloudError,
    CloudUnavailableError,
    EmailConfirmationRequiredError,
    InvalidCredentialsError,
    RateLimitError,
    UserAlreadyExistsError,
    WeakPasswordError,
)
from app.cloud.config import CloudConfig, load_cloud_config

CONFIG = CloudConfig(url="https://demo.supabase.co", anon_key="anon")


class FakeResponse:
    def __init__(self, status: int, body: object = None) -> None:
        self.status_code = status
        self._body = body
        self.content = b"" if body is None else b"x"
        self.text = ""

    def json(self) -> object:
        if self._body is None:
            raise ValueError("no body")
        return self._body


class FakeHttp:
    """Returns queued responses and records every request."""

    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def request(self, method: str, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


class BrokenHttp:
    def request(self, *args: object, **kwargs: object) -> None:
        raise requests.ConnectionError("no network")


def auth_body(token: str = "access", refresh: str = "refresh") -> dict:
    return {
        "access_token": token,
        "refresh_token": refresh,
        "user": {
            "id": "user-1",
            "email": "a@b.ru",
            "user_metadata": {"first_name": "Анна", "last_name": "Иванова"},
        },
    }


def test_sign_in_stores_session_and_names() -> None:
    http = FakeHttp(FakeResponse(200, auth_body()))
    client = CloudClient(CONFIG, http)
    session = client.sign_in("a@b.ru", "secret1")
    assert session.first_name == "Анна"
    assert client.session == session
    call = http.calls[0]
    assert call["url"] == "https://demo.supabase.co/auth/v1/token"
    assert call["params"] == {"grant_type": "password"}


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (400, {"error_code": "invalid_credentials", "msg": "Invalid login credentials"},
         InvalidCredentialsError),
        (400, {"error": "invalid_grant", "error_description": "Invalid login credentials"},
         InvalidCredentialsError),
        (422, {"error_code": "user_already_exists", "msg": "User already registered"},
         UserAlreadyExistsError),
        (422, {"error_code": "weak_password", "msg": "weak"}, WeakPasswordError),
        (429, {"error_code": "over_email_send_rate_limit", "msg": "email rate limit exceeded"},
         RateLimitError),
        (503, {"message": "down"}, CloudUnavailableError),
        (418, {"message": "teapot"}, CloudError),
    ],
)
def test_error_responses_become_exceptions(status: int, body: dict, error: type) -> None:
    client = CloudClient(CONFIG, FakeHttp(FakeResponse(status, body)))
    with pytest.raises(error):
        client.sign_in("a@b.ru", "wrong")


def test_network_failure_means_unavailable() -> None:
    with pytest.raises(CloudUnavailableError):
        CloudClient(CONFIG, BrokenHttp()).sign_in("a@b.ru", "secret1")


def test_sign_up_without_session_requires_confirmation() -> None:
    client = CloudClient(CONFIG, FakeHttp(FakeResponse(200, {"id": "user-1"})))
    with pytest.raises(EmailConfirmationRequiredError):
        client.sign_up("a@b.ru", "secret1", "Анна", "Иванова")


def test_expired_token_is_refreshed_once() -> None:
    http = FakeHttp(
        FakeResponse(200, auth_body("old", "r1")),
        FakeResponse(401, {"message": "JWT expired"}),
        FakeResponse(200, auth_body("new", "r2")),
        FakeResponse(200, [{"id": "s1"}]),
    )
    client = CloudClient(CONFIG, http)
    client.sign_in("a@b.ru", "secret1")
    assert client.select("sessions", {"select": "*"}) == [{"id": "s1"}]
    assert http.calls[-1]["headers"]["Authorization"] == "Bearer new"
    assert client.session.refresh_token == "r2"


def test_delete_uses_in_filter() -> None:
    http = FakeHttp(FakeResponse(200, auth_body()), FakeResponse(204))
    client = CloudClient(CONFIG, http)
    client.sign_in("a@b.ru", "secret1")
    client.delete("sessions", ["s1", "s2"])
    assert http.calls[-1]["params"] == {"id": "in.(s1,s2)"}


def test_config_file(tmp_path) -> None:
    path = tmp_path / "cloud.json"
    assert load_cloud_config(path) is None
    path.write_text('{"url": "https://x.supabase.co/", "anon_key": "k"}')
    assert load_cloud_config(path) == CloudConfig("https://x.supabase.co", "k")


def test_auth_requests_send_only_the_project_key() -> None:
    http = FakeHttp(FakeResponse(200, auth_body()))
    CloudClient(CONFIG, http).sign_in("a@b.ru", "secret1")
    headers = http.calls[0]["headers"]
    assert headers["apikey"] == "anon"
    assert "Authorization" not in headers
