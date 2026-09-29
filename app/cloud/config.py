"""Connection settings of the cloud backend (Supabase).

They are read from ~/.focuscore/cloud.json, so keys never get into the
repository. Expected content:
    {"url": "https://<project>.supabase.co", "anon_key": "<public anon key>"}
"""

import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path.home() / ".focuscore" / "cloud.json"


@dataclass(frozen=True)
class CloudConfig:
    """Address of the Supabase project and its public (anon) API key."""

    url: str
    anon_key: str


def load_cloud_config(path: Path = CONFIG_PATH) -> CloudConfig | None:
    """Return the configuration, or None if the file is missing or invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CloudConfig(url=data["url"].rstrip("/"), anon_key=data["anon_key"])
    except (OSError, ValueError, KeyError, AttributeError):
        return None
