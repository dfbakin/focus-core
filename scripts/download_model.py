"""Download the engagement model from the GitHub release of the project.

The model (about 130 MB) is too large for the git repository, so it is
attached to the release. Run once after cloning, from the repository root:
    python -m scripts.download_model
"""

import sys

import requests

from app.core.engagement import MODEL_PATH

MODEL_URL = (
    "https://github.com/Ayanami-Rei1/FocusCore/releases/download/v1.0/engagement_slow_r50.ts"
)
CHUNK_BYTES = 1 << 20
TIMEOUT_SEC = 60


def download(url: str = MODEL_URL) -> None:
    """Save the model to MODEL_PATH, showing the progress."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = MODEL_PATH.with_suffix(".part")
    with requests.get(url, stream=True, timeout=TIMEOUT_SEC) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        received = 0
        with partial.open("wb") as file:
            for chunk in response.iter_content(CHUNK_BYTES):
                file.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\r{received * 100 // total:3d} %", end="", flush=True)
    partial.replace(MODEL_PATH)
    print(f"\nSaved {MODEL_PATH}")


def main() -> None:
    """Download the model unless it is already there."""
    if MODEL_PATH.exists():
        print(f"Model is already in place: {MODEL_PATH}")
        return
    try:
        download()
    except requests.RequestException as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
