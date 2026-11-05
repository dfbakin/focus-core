"""Audio sources: the microphone and the audio track of a video file.

sounddevice loads the PortAudio library as soon as it is imported, so it is
imported inside the functions that need it. The rest of the application and
the tests then also work on machines without PortAudio.
"""

from pathlib import Path

import av
import numpy as np

from app.core.audio import SAMPLE_RATE, AudioBuffer

MICROPHONE_CHECK_MS = 200


class AudioSourceError(Exception):
    """Base error for an audio source that is not ready."""


class MicrophoneUnavailableError(AudioSourceError):
    """There is no microphone, or it cannot be opened."""


class AudioTrackMissingError(AudioSourceError):
    """The video file has no audio track or it cannot be decoded."""


def _sounddevice():
    """Import sounddevice, turning a missing PortAudio into our error."""
    try:
        import sounddevice
    except OSError as exc:
        raise MicrophoneUnavailableError("PortAudio is not available") from exc
    return sounddevice


def check_microphone() -> str:
    """Open the default microphone for a moment and return its name.

    Raises:
        MicrophoneUnavailableError: No input device, or it cannot be opened.
    """
    sd = _sounddevice()
    try:
        device = sd.query_devices(kind="input")
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16"):
            sd.sleep(MICROPHONE_CHECK_MS)
    except (sd.PortAudioError, ValueError) as exc:
        raise MicrophoneUnavailableError(str(exc)) from exc
    return device["name"]


class MicrophoneRecorder:
    """Records the default microphone into an AudioBuffer.

    PortAudio calls `_on_audio` from its own thread with every new chunk.
    While paused, chunks are dropped, so the recording stays aligned with
    the active time of the lecture.
    """

    def __init__(self, buffer: AudioBuffer) -> None:
        self._buffer = buffer
        self._stream = None
        self._paused = False

    def start(self) -> None:
        """Start recording.

        Raises:
            MicrophoneUnavailableError: The microphone cannot be opened.
        """
        sd = _sounddevice()
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                callback=self._on_audio,
            )
            self._stream.start()
        except (sd.PortAudioError, ValueError) as exc:
            self._stream = None
            raise MicrophoneUnavailableError(str(exc)) from exc

    def set_paused(self, paused: bool) -> None:
        """Pause or resume recording."""
        self._paused = paused

    def stop(self) -> None:
        """Stop recording and release the microphone."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Store a chunk from the microphone unless recording is paused."""
        if not self._paused:
            self._buffer.append(indata[:, 0].copy())


def check_audio_track(path: str | Path) -> None:
    """Make sure the video file contains an audio track.

    Raises:
        AudioTrackMissingError: No audio track, or the file cannot be read.
    """
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioTrackMissingError(f"No audio track in {path}")
    except av.FFmpegError as exc:
        raise AudioTrackMissingError(str(exc)) from exc


def read_audio_track(path: str | Path) -> AudioBuffer:
    """Decode the audio track of a video file into an AudioBuffer.

    Any sample rate and channel layout is converted to 16 kHz mono.

    Raises:
        AudioTrackMissingError: No audio track, or the file cannot be read.
    """
    buffer = AudioBuffer()
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioTrackMissingError(f"No audio track in {path}")
            for frame in container.decode(container.streams.audio[0]):
                for converted in resampler.resample(frame):
                    buffer.append(converted.to_ndarray())
        for converted in resampler.resample(None):
            buffer.append(converted.to_ndarray())
    except av.FFmpegError as exc:
        raise AudioTrackMissingError(str(exc)) from exc
    return buffer
