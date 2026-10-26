"""Hardware resource checks performed before a lecture starts."""

from dataclasses import dataclass

import psutil

DEFAULT_MIN_FREE_RAM_MB = 1024
BYTES_IN_MB = 1024 * 1024


@dataclass(frozen=True)
class MemoryStatus:
    """Free RAM compared with the amount required for analysis."""

    available_mb: int
    required_mb: int

    @property
    def is_enough(self) -> bool:
        """True if analysis can be started."""
        return self.available_mb >= self.required_mb


def check_memory(required_mb: int) -> MemoryStatus:
    """Measure the RAM currently available to new processes."""
    available_mb = psutil.virtual_memory().available // BYTES_IN_MB
    return MemoryStatus(available_mb=available_mb, required_mb=required_mb)
