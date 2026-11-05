"""Tests for transcription of critical moments, with a fake OpenAI endpoint."""

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest
import requests

from app.core.audio import AudioBuffer
from app.core.report import CriticalMoment
from app.core.transcription import (
    InvalidKeyError,
    QuotaExceededError,
    RegionBlockedError,
    ServiceUnavailableError,
    WhisperClient,
    encode_wav,
    load_api_key,
    excerpt_range,
    transcribe_moments,
)


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> dict:
        return self._body


class FakePost:
    """Records requests and answers with a prepared response."""

    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def __call__(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


def test_wav_keeps_samples_and_rate() -> None:
    samples = np.arange(-50, 50, dtype=np.int16)
    with wave.open(io.BytesIO(encode_wav(samples, 16000))) as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        restored = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    assert restored.tolist() == samples.tolist()


def test_excerpt_is_mostly_before_the_drop() -> None:
    assert excerpt_range(100.0) == (75.0, 105.0)


def test_excerpt_does_not_start_before_the_lecture() -> None:
    assert excerpt_range(10.0) == (0.0, 15.0)


def test_client_sends_key_model_and_language() -> None:
    post = FakePost(FakeResponse(200, {"text": " Сегодня разберём производные. "}))
    text = WhisperClient("sk-test", post).transcribe(np.zeros(16000, np.int16), 16000)
    assert text == "Сегодня разберём производные."
    call = post.calls[0]
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["data"] == {"model": "whisper-1", "language": "ru"}
    assert call["files"]["file"][0] == "moment.wav"


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (401, {"error": {"message": "Incorrect API key"}}, InvalidKeyError),
        (403, {"error": {"code": "unsupported_country_region_territory"}}, RegionBlockedError),
        (429, {"error": {"code": "insufficient_quota"}}, QuotaExceededError),
        (500, {}, ServiceUnavailableError),
    ],
)
def test_errors_are_recognized(status: int, body: dict, error: type) -> None:
    client = WhisperClient("sk-test", FakePost(FakeResponse(status, body)))
    with pytest.raises(error):
        client.transcribe(np.zeros(16000, np.int16), 16000)


def test_network_failure_means_service_unavailable() -> None:
    def failing_post(url: str, **kwargs):
        raise requests.ConnectionError("no network")

    with pytest.raises(ServiceUnavailableError):
        WhisperClient("sk-test", failing_post).transcribe(np.zeros(16000, np.int16), 16000)


def test_every_drop_with_audio_is_transcribed() -> None:
    audio = AudioBuffer(sample_rate=100)
    audio.append(np.ones(100 * 200, np.int16))
    post = FakePost(FakeResponse(200, {"text": "текст"}))
    points = [(30.0, 60.0), (40.0, 20.0), (60.0, 35.0), (120.0, 10.0), (130.0, 60.0),
              (300.0, 60.0), (310.0, 20.0), (320.0, 60.0)]
    moments = [CriticalMoment(40.0, 130.0, 10.0), CriticalMoment(310.0, 320.0, 20.0)]
    texts = transcribe_moments(moments, points, audio, WhisperClient("sk-test", post))
    assert texts == ["текст\n\nтекст", None]
    assert len(post.calls) == 2


def test_key_is_read_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / "openai.json"
    assert load_api_key(path) is None
    path.write_text(json.dumps({"api_key": " sk-file "}))
    assert load_api_key(path) == "sk-file"
