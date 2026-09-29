"""Minimal client for the Supabase REST API: authentication and table access.

Talks to two Supabase services over HTTPS:
    /auth/v1  sign-up, sign-in, token refresh and user updates;
    /rest/v1  reading and writing the sessions and engagement_points tables.
Row Level Security on the server guarantees that a user sees only own rows.
"""

from dataclasses import dataclass, replace

import requests

from app.cloud.config import CloudConfig

REQUEST_TIMEOUT_SEC = 10
UNAUTHORIZED = 401
SERVER_ERROR = 500


class CloudError(Exception):
    """Any failure reported by the cloud backend."""


class CloudUnavailableError(CloudError):
    """The server cannot be reached: no internet, server down or not configured."""


class UserAlreadyExistsError(CloudError):
    """Registration with an email that is already registered."""


class InvalidCredentialsError(CloudError):
    """Wrong email or password, or an expired sign-in."""


class WeakPasswordError(CloudError):
    """The server rejected the password as too simple."""


class RateLimitError(CloudError):
    """Too many requests of the same kind in a short time."""


class EmailConfirmationRequiredError(CloudError):
    """The project requires confirming the email before the first sign-in."""


@dataclass(frozen=True)
class AuthSession:
    """A signed-in user with the tokens that authorize requests."""

    user_id: str
    email: str
    first_name: str
    last_name: str
    access_token: str
    refresh_token: str


class CloudClient:
    """Sends HTTP requests to Supabase and turns error responses into exceptions.

    Args:
        config: Project address and public key.
        http: Object with a requests-compatible `request` method; tests pass
            a fake one, the application uses requests.Session.
    """

    def __init__(self, config: CloudConfig, http: requests.Session | None = None) -> None:
        self._config = config
        self._http = http or requests.Session()
        self.session: AuthSession | None = None

    def sign_up(self, email: str, password: str, first_name: str, last_name: str) -> AuthSession:
        """Register a new user and sign in."""
        data = self._auth_request(
            "/auth/v1/signup",
            {
                "email": email,
                "password": password,
                "data": {"first_name": first_name, "last_name": last_name},
            },
        )
        if "access_token" not in data:
            raise EmailConfirmationRequiredError("Email confirmation is enabled")
        return self._remember(data)

    def sign_in(self, email: str, password: str) -> AuthSession:
        """Sign in with email and password."""
        data = self._auth_request(
            "/auth/v1/token",
            {"email": email, "password": password},
            params={"grant_type": "password"},
        )
        return self._remember(data)

    def restore(self, refresh_token: str) -> AuthSession:
        """Get fresh tokens from a refresh token saved at the previous launch."""
        data = self._auth_request(
            "/auth/v1/token",
            {"refresh_token": refresh_token},
            params={"grant_type": "refresh_token"},
        )
        return self._remember(data)

    def update_profile(self, first_name: str, last_name: str) -> AuthSession:
        """Change the first and last name stored in the user's metadata."""
        self._request(
            "PUT", "/auth/v1/user", json={"data": {"first_name": first_name, "last_name": last_name}}
        )
        self.session = replace(self._signed_in(), first_name=first_name, last_name=last_name)
        return self.session

    def change_password(self, new_password: str) -> None:
        """Set a new password for the signed-in user."""
        self._request("PUT", "/auth/v1/user", json={"password": new_password})

    def select(self, table: str, params: dict[str, str]) -> list[dict]:
        """Read rows of a table; `params` are PostgREST filters."""
        return self._request("GET", f"/rest/v1/{table}", params=params) or []

    def upsert(self, table: str, rows: list[dict]) -> None:
        """Insert rows, overwriting existing ones with the same id."""
        if rows:
            self._request(
                "POST",
                f"/rest/v1/{table}",
                json=rows,
                params={"on_conflict": "id"},
                extra_headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            )

    def delete(self, table: str, ids: list[str]) -> None:
        """Delete rows with the given ids."""
        if ids:
            self._request("DELETE", f"/rest/v1/{table}", params={"id": f"in.({','.join(ids)})"})

    def _signed_in(self) -> AuthSession:
        """Return the current session or fail if nobody is signed in."""
        if self.session is None:
            raise InvalidCredentialsError("Not signed in")
        return self.session

    def _auth_request(self, path: str, body: dict, params: dict | None = None) -> dict:
        """POST to the authentication service without a user token."""
        response = self._send("POST", path, json=body, params=params, token=None)
        self._raise_for_error(response)
        return response.json()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: object = None,
        params: dict | None = None,
        extra_headers: dict | None = None,
    ) -> object:
        """Authorized request; if the access token expired, refresh it once and retry."""
        session = self._signed_in()
        response = self._send(
            method, path, json=json, params=params, token=session.access_token,
            extra_headers=extra_headers,
        )
        if response.status_code == UNAUTHORIZED:
            session = self.restore(session.refresh_token)
            response = self._send(
                method, path, json=json, params=params, token=session.access_token,
                extra_headers=extra_headers,
            )
        self._raise_for_error(response)
        return response.json() if response.content else None

    def _send(
        self,
        method: str,
        path: str,
        *,
        json: object,
        params: dict | None,
        token: str | None,
        extra_headers: dict | None = None,
    ) -> requests.Response:
        """Perform the HTTP request; network problems become CloudUnavailableError.

        The project key always goes in the apikey header. The Authorization
        header is added only for a signed-in user, because the newer
        publishable keys are not tokens and must not be sent there.
        """
        headers = {
            "apikey": self._config.anon_key,
            "Content-Type": "application/json",
            **(extra_headers or {}),
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            return self._http.request(
                method,
                self._config.url + path,
                json=json,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SEC,
            )
        except requests.RequestException as exc:
            raise CloudUnavailableError(str(exc)) from exc

    def _remember(self, data: dict) -> AuthSession:
        """Build the session from an authentication response and keep it."""
        user = data["user"]
        metadata = user.get("user_metadata") or {}
        self.session = AuthSession(
            user_id=user["id"],
            email=user.get("email", ""),
            first_name=metadata.get("first_name", ""),
            last_name=metadata.get("last_name", ""),
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
        )
        return self.session

    @staticmethod
    def _raise_for_error(response: requests.Response) -> None:
        """Translate an error response into the matching exception."""
        if response.status_code < 400:
            return
        try:
            body = response.json()
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        code = str(body.get("error_code") or body.get("error") or "")
        message = str(
            body.get("msg") or body.get("error_description") or body.get("message") or ""
        )
        if code == "user_already_exists" or "already registered" in message:
            raise UserAlreadyExistsError(message)
        if code in ("invalid_credentials", "invalid_grant", "refresh_token_not_found"):
            raise InvalidCredentialsError(message)
        if code == "weak_password":
            raise WeakPasswordError(message)
        if "rate_limit" in code or "rate limit" in message or response.status_code == 429:
            raise RateLimitError(message)
        if response.status_code >= SERVER_ERROR:
            raise CloudUnavailableError(message or f"HTTP {response.status_code}")
        raise CloudError(message or f"HTTP {response.status_code}")
