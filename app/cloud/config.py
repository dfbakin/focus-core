"""Connection settings of the cloud backend (Supabase).

The application ships with the address and the public key of the FocusCore
project in default_config.json, so every user gets accounts and
synchronization right after installation. The public key is meant to be
shared: Row Level Security on the server lets each user reach only own rows.
A developer can point the application to another project with
~/.focuscore/cloud.json, which takes priority. Expected content of both:
    {"url": "https://<project>.supabase.co", "anon_key": "<public key>"}
"""

import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path.home() / ".focuscore" / "cloud.json"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.json"


@dataclass(frozen=True)
class CloudConfig:
    """Address of the Supabase project and its public (anon) API key."""

    url: str
    anon_key: str


def load_cloud_config(
    path: Path = CONFIG_PATH, default_path: Path = DEFAULT_CONFIG_PATH
) -> CloudConfig | None:
    """Return the user's configuration, else the bundled one, else None."""
    return _read_config(path) or _read_config(default_path)


def _read_config(path: Path) -> CloudConfig | None:
    """Read one configuration file; None if it is missing or invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CloudConfig(url=data["url"].rstrip("/"), anon_key=data["anon_key"])
    except (OSError, ValueError, KeyError, AttributeError):
        return None
