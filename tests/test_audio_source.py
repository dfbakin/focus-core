"""Tests for reading the audio track of a video file."""

from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from app.core.audio import SAMPLE_RATE
from app.core.audio_source import (
    AudioTrackMissingError,
    check_audio_track,
    read_audio_track,
)


def _write_stereo_wav(path: Path, seconds: float, rate: int = 44100) -> None:
    """Write a stereo 16-bit WAV file with a quiet tone."""
    samples = int(seconds * rate)
    tone = (np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, samples)) * 3000).astype(np.int16)
    stereo = np.stack([tone, tone]).reshape(1, -1, order="F")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=rate, layout="stereo")
        frame = av.AudioFrame.from_ndarray(stereo, format="s16", layout="stereo")
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def _write_silent_video(path: Path) -> None:
    """Write a short video without an audio track."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(10):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


def test_audio_track_is_converted_to_16k_mono(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    _write_stereo_wav(path, seconds=1.5)
    check_audio_track(path)
    buffer = read_audio_track(path)
    assert buffer.sample_rate == SAMPLE_RATE
    assert buffer.duration_sec == pytest.approx(1.5, abs=0.05)
    assert np.abs(buffer.clip(0.0, 1.0)).max() > 0


def test_video_without_audio_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "silent.mp4"
    _write_silent_video(path)
    with pytest.raises(AudioTrackMissingError):
        check_audio_track(path)


def test_unreadable_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.mp4"
    path.write_text("not a video")
    with pytest.raises(AudioTrackMissingError):
        check_audio_track(path)
