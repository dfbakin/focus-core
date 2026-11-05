"""Speech transcription of critical moments with the OpenAI Whisper API.

For every critical moment only short fragments are sent: about half a
minute of speech around each of its deepest drops of engagement (see
report.find_dips). The rest of the lecture never leaves the computer, and
the recording is dropped right after (see transcription_job). The API key
is read from ~/.focuscore/openai.json, so it never gets into the repository:
    {"api_key": "sk-..."}
"""

import io
import json
import os
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import requests

from app.core.audio import AudioBuffer
from app.core.report import TRANSCRIPT_SEPARATOR, CriticalMoment, find_dips

KEY_PATH = Path.home() / ".focuscore" / "openai.json"
KEY_ENV_VARIABLE = "OPENAI_API_KEY"
API_URL = "https://api.openai.com/v1/audio/transcriptions"
MODEL = "whisper-1"
LANGUAGE = "ru"
REQUEST_TIMEOUT_SEC = 120
SECONDS_BEFORE_DIP = 25.0
SECONDS_AFTER_DIP = 5.0
MIN_CLIP_SEC = 1.0
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_TOO_MANY_REQUESTS = 429


class TranscriptionError(Exception):
    """Base error: the audio could not be transcribed."""


class InvalidKeyError(TranscriptionError):
    """The API key is missing or rejected."""


class RegionBlockedError(TranscriptionError):
    """The service is not available from the current country without a VPN."""


class QuotaExceededError(TranscriptionError):
    """The account has no money left or too many requests were sent."""


class ServiceUnavailableError(TranscriptionError):
    """No connection to the service, or it failed on its side."""


def load_api_key(path: Path = KEY_PATH) -> str | None:
    """Return the key from the environment or the key file, or None."""
    key = os.environ.get(KEY_ENV_VARIABLE)
    if key:
        return key.strip()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data["api_key"]).strip() or None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Pack 16-bit mono samples into a WAV file in memory."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return output.getvalue()


def excerpt_range(dip_sec: float) -> tuple[float, float]:
    """Return the half minute of speech that explains a drop of engagement.

    Engagement reacts to what was said a little earlier, and every estimate
    already covers about two seconds of video, so most of the fragment lies
    before the lowest point.
    """
    return max(0.0, dip_sec - SECONDS_BEFORE_DIP), dip_sec + SECONDS_AFTER_DIP


class WhisperClient:
    """Sends one audio clip at a time to the transcription endpoint.

    Args:
        api_key: Secret key of the OpenAI account.
        post: Function with the signature of requests.post; tests pass a fake.
    """

    def __init__(self, api_key: str, post: Callable = requests.post) -> None:
        self._api_key = api_key
        self._post = post

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        """Return the recognized text of the clip.

        Raises:
            TranscriptionError: A subclass that tells what went wrong.
        """
        try:
            response = self._post(
                API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                data={"model": MODEL, "language": LANGUAGE},
                files={"file": ("moment.wav", encode_wav(samples, sample_rate), "audio/wav")},
                timeout=REQUEST_TIMEOUT_SEC,
            )
        except requests.RequestException as exc:
            raise ServiceUnavailableError(str(exc)) from exc
        if response.status_code != 200:
            raise _error_for(response)
        return response.json().get("text", "").strip()


def _error_for(response: requests.Response) -> TranscriptionError:
    """Turn an error response of the API into the matching exception."""
    try:
        error = response.json().get("error", {})
    except ValueError:
        error = {}
    code = str(error.get("code") or "")
    message = str(error.get("message") or response.status_code)
    if response.status_code == HTTP_UNAUTHORIZED:
        return InvalidKeyError(message)
    if response.status_code == HTTP_FORBIDDEN and "region" in code + message:
        return RegionBlockedError(message)
    if response.status_code == HTTP_TOO_MANY_REQUESTS:
        return QuotaExceededError(message)
    return ServiceUnavailableError(message)


def transcribe_moments(
    moments: list[CriticalMoment],
    points: list[tuple[float, float]],
    audio: AudioBuffer,
    client,
) -> list[str | None]:
    """Transcribe the speech around the drops of every moment, in the same order.

    `client` is anything with a `transcribe(samples, sample_rate)` method:
    a WhisperClient or the user's account. Fragments of one moment are
    joined with TRANSCRIPT_SEPARATOR. A moment without audio, for example
    one that happened during a pause, gets None.

    Raises:
        TranscriptionError: The service failed; nothing is returned then.
    """
    texts: list[str | None] = []
    for moment in moments:
        parts = []
        for dip in find_dips(points, moment):
            clip = audio.clip(*excerpt_range(dip))
            if len(clip) < MIN_CLIP_SEC * audio.sample_rate:
                continue
            text = client.transcribe(clip, audio.sample_rate)
            if text:
                parts.append(text)
        texts.append(TRANSCRIPT_SEPARATOR.join(parts) or None)
    return texts
