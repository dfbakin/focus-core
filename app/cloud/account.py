"""Account operations of user scenario 10.

Combines the cloud client with the local database: remembers the signed-in
account between launches and runs synchronization after every change.
All methods are blocking; the interface calls them from a background thread.
"""

from pathlib import Path

from app.cloud.client import (
    AuthSession,
    CloudClient,
    CloudUnavailableError,
    InvalidCredentialsError,
)
from app.cloud.config import load_cloud_config
from app.cloud.sync import SyncResult, synchronize
from app.db.account import StoredAccount, clear_account, load_account, save_account
from app.db.local import DB_PATH, LOCAL_USER_ID
from app.db.sessions import reassign_sessions


class AccountService:
    """Sign-up, sign-in, profile changes and synchronization.

    Args:
        client: Configured cloud client, or None if the cloud is not set up;
            then every network operation raises CloudUnavailableError and the
            application keeps working offline.
    """

    def __init__(self, client: CloudClient | None, db_path: Path = DB_PATH) -> None:
        self._client = client
        self._db_path = db_path

    @property
    def is_configured(self) -> bool:
        """False if ~/.focuscore/cloud.json is missing: the app works offline."""
        return self._client is not None

    @property
    def account(self) -> StoredAccount | None:
        """Account remembered on this device."""
        return load_account(self._db_path)

    def register(self, first_name: str, last_name: str, email: str, password: str) -> SyncResult:
        """Create an account, sign in and synchronize."""
        session = self._cloud().sign_up(email, password, first_name, last_name)
        return self._after_sign_in(session)

    def sign_in(self, email: str, password: str) -> SyncResult:
        """Sign in and download the user's sessions."""
        session = self._cloud().sign_in(email, password)
        return self._after_sign_in(session)

    def restore(self) -> SyncResult | None:
        """At startup, sign in again with the saved token and synchronize.

        Returns None if nobody was signed in. If the saved sign-in expired,
        the account is forgotten and InvalidCredentialsError is raised.
        """
        account = self.account
        if account is None:
            return None
        try:
            self._cloud().restore(account.refresh_token)
        except InvalidCredentialsError:
            clear_account(self._db_path)
            raise
        self._save_tokens()
        return self.synchronize()

    def synchronize(self) -> SyncResult:
        """Exchange sessions with the server."""
        account = self._require_account()
        client = self._cloud()
        if client.session is None:
            client.restore(account.refresh_token)
        result = synchronize(client, account.user_id, self._db_path)
        self._save_tokens()
        return result

    def update_profile(self, first_name: str, last_name: str) -> None:
        """Change the name on the server and locally."""
        self._ensure_signed_in()
        self._cloud().update_profile(first_name, last_name)
        self._save_tokens()

    def change_password(self, current_password: str, new_password: str) -> None:
        """Check the current password by signing in with it, then set the new one."""
        account = self._require_account()
        client = self._cloud()
        client.sign_in(account.email, current_password)
        client.change_password(new_password)
        self._save_tokens()

    def sign_out(self) -> None:
        """Forget the account on this device; sessions stay in the local database."""
        clear_account(self._db_path)
        if self._client is not None:
            self._client.session = None

    def _after_sign_in(self, session: AuthSession) -> SyncResult:
        """Remember the account, adopt sessions recorded before sign-in, synchronize."""
        self._save_tokens()
        reassign_sessions(LOCAL_USER_ID, session.user_id, self._db_path)
        return self.synchronize()

    def _save_tokens(self) -> None:
        """Persist the current session; refresh tokens change after every use."""
        session = self._cloud().session
        if session is None:
            return
        save_account(
            StoredAccount(
                user_id=session.user_id,
                email=session.email,
                first_name=session.first_name,
                last_name=session.last_name,
                refresh_token=session.refresh_token,
            ),
            self._db_path,
        )

    def _ensure_signed_in(self) -> None:
        """Restore the server session from the saved token if needed."""
        account = self._require_account()
        if self._cloud().session is None:
            self._cloud().restore(account.refresh_token)

    def _require_account(self) -> StoredAccount:
        """Return the stored account or fail if nobody is signed in."""
        account = self.account
        if account is None:
            raise InvalidCredentialsError("Not signed in")
        return account

    def _cloud(self) -> CloudClient:
        """Return the client or report that the cloud is not configured."""
        if self._client is None:
            raise CloudUnavailableError("Cloud storage is not configured")
        return self._client


def create_account_service(db_path: Path = DB_PATH) -> AccountService:
    """Build the service from ~/.focuscore/cloud.json, working offline without it."""
    config = load_cloud_config()
    client = CloudClient(config) if config else None
    return AccountService(client, db_path)
