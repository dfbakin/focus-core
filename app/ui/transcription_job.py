"""Background transcription of a finished lecture (user scenario 6).

Ties together the audio, a transcriber and the database, so the lecture page
only has to start the job and show its result. The transcriber is the
user's account (speech goes through the project's Edge Function, which
keeps the OpenAI key on the server) or, for development, a WhisperClient
with a personal key.
"""

from functools import partial
from pathlib import Path
from typing import Protocol

import numpy as np

from app.cloud.client import (
    CloudUnavailableError,
    InvalidCredentialsError,
    RateLimitError,
)
from app.core.audio import AudioBuffer
from app.core.audio_source import AudioTrackMissingError, read_audio_track
from app.core.report import CriticalMoment
from app.core.transcription import (
    InvalidKeyError,
    QuotaExceededError,
    RegionBlockedError,
    transcribe_moments,
)
from app.db.moments import save_transcripts
from app.ui.workers import TaskWorker

NO_TRANSCRIBER_TEXT = "Расшифровка не выполнена: войдите в аккаунт в разделе «Профиль»."
SIGNED_OUT_TEXT = "Расшифровка не выполнена: вход в аккаунт истёк, войдите снова."
INVALID_KEY_TEXT = "Расшифровка не выполнена: ключ OpenAI недействителен."
REGION_TEXT = (
    "Расшифровка не выполнена: сервис OpenAI недоступен из вашего региона. "
    "Включите VPN перед следующей лекцией."
)
QUOTA_TEXT = (
    "Расшифровка не выполнена: исчерпан лимит сервиса расшифровки. "
    "Попробуйте позже."
)
NO_AUDIO_TEXT = "Расшифровка не выполнена: не удалось прочитать звуковую дорожку."
CONNECTION_TEXT = "Расшифровка не выполнена: нет соединения с сервером. Проверьте интернет."


class Transcriber(Protocol):
    """Anything that turns a clip of 16-bit mono samples into text."""

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        """Return the recognized speech."""


def create_transcription_worker(
    session_id: str,
    moments: list[CriticalMoment],
    points: list[tuple[float, float]],
    audio: AudioBuffer | None,
    audio_path: str | None,
    transcriber: Transcriber,
    db_path: Path,
    parent,
) -> TaskWorker | None:
    """Create a background worker that transcribes the moments.

    The caller connects its signals and starts it. The recording is handed
    over to the job and cleared when the job ends, whatever the outcome.
    Returns None, after clearing the recording, if there is nothing to
    transcribe.
    """
    if not moments:
        forget(audio)
        return None
    job = partial(
        _transcribe_and_store, session_id, moments, points, audio, audio_path,
        transcriber, db_path,
    )
    return TaskWorker(job, parent)


def _transcribe_and_store(
    session_id: str,
    moments: list[CriticalMoment],
    points: list[tuple[float, float]],
    audio: AudioBuffer | None,
    audio_path: str | None,
    transcriber: Transcriber,
    db_path: Path,
) -> None:
    """Thread body: get the audio, transcribe every moment, save the texts."""
    try:
        if audio is None:
            audio = read_audio_track(audio_path)
        texts = transcribe_moments(moments, points, audio, transcriber)
        save_transcripts(session_id, moments, texts, db_path)
    finally:
        forget(audio)


def forget(audio: AudioBuffer | None) -> None:
    """Clear the recording, so no sound of the lecture stays in memory."""
    if audio is not None:
        audio.clear()


def error_text(error: Exception) -> str:
    """Explain to the user why the transcription failed."""
    messages = (
        (InvalidCredentialsError, SIGNED_OUT_TEXT),
        (InvalidKeyError, INVALID_KEY_TEXT),
        (RegionBlockedError, REGION_TEXT),
        ((QuotaExceededError, RateLimitError), QUOTA_TEXT),
        (AudioTrackMissingError, NO_AUDIO_TEXT),
        (CloudUnavailableError, CONNECTION_TEXT),
    )
    for error_type, text in messages:
        if isinstance(error, error_type):
            return text
    return CONNECTION_TEXT
