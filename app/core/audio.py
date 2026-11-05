"""Audio kept in memory for the duration of a lecture.

The recording is never written to disk: it lives in an AudioBuffer until the
critical moments are transcribed and is dropped right after. Pure Python and
numpy, so the buffer is tested without a microphone.
"""

import threading

import numpy as np

SAMPLE_RATE = 16000


class AudioBuffer:
    """Mono 16-bit audio at SAMPLE_RATE, collected chunk by chunk.

    16 kHz mono is the format speech recognition expects, and it takes about
    115 MB per hour. Chunks may be appended from the recording thread while
    the interface reads the duration, so access is guarded by a lock.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._length = 0
        self._lock = threading.Lock()

    def append(self, samples: np.ndarray) -> None:
        """Add samples at the end of the recording."""
        chunk = np.asarray(samples, dtype=np.int16).ravel()
        with self._lock:
            self._chunks.append(chunk)
            self._length += len(chunk)

    @property
    def duration_sec(self) -> float:
        """Length of the recording in seconds."""
        with self._lock:
            return self._length / self.sample_rate

    def clip(self, start_sec: float, end_sec: float) -> np.ndarray:
        """Return the samples between two moments, clamped to the recording."""
        with self._lock:
            samples = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.int16)
        start = max(0, int(start_sec * self.sample_rate))
        end = min(len(samples), int(end_sec * self.sample_rate))
        return samples[start:end] if end > start else np.zeros(0, np.int16)

    def clear(self) -> None:
        """Forget the recording."""
        with self._lock:
            self._chunks.clear()
            self._length = 0
