"""Download speeds over a sliding window, measured from progress snapshots.

Used only by the command line display.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from .progress import DownloadProgress, FileState

WINDOW_SECONDS = 10.0
MIN_SPAN_SECONDS = 2.0  # less data than this gives no speed yet
MEASURED_STATES = (FileState.DOWNLOADING, FileState.RETRYING)


class SpeedMeter:
    """The speed of a growing byte count over the last few seconds."""

    def __init__(self, window: float = WINDOW_SECONDS, min_span: float = MIN_SPAN_SECONDS) -> None:
        """Create the meter.

        Args:
            window: How many seconds of samples the speed covers.
            min_span: The fewest seconds of samples needed to give a speed.
        """
        self._window = window
        self._min_span = min_span
        self._samples: deque[tuple[float, int]] = deque()

    def add(self, timestamp: float, value: int) -> float | None:
        """Record a sample and return the current speed.

        A value smaller than the previous one (a file that started again) clears the
        samples, so the speed never goes negative.

        Args:
            timestamp: When the value was measured, in seconds.
            value: The byte count at that time.

        Returns:
            The bytes per second over the window, or None until there is enough data.
        """
        if self._samples and value < self._samples[-1][1]:
            self._samples.clear()
        self._samples.append((timestamp, value))
        # Keep one sample at or before the window start, so the window is fully covered.
        while len(self._samples) > 2 and self._samples[1][0] <= timestamp - self._window:
            self._samples.popleft()
        first_time, first_value = self._samples[0]
        span = timestamp - first_time
        return (value - first_value) / span if span >= self._min_span else None


@dataclass(frozen=True)
class Stats:
    """What the display measured, as of one snapshot.

    Attributes:
        speed: The total speed in bytes per second, or None until there is enough data.
        file_speeds: The speed of each file that downloads, by repo path.
        elapsed: Seconds since the first snapshot.
    """

    speed: float | None
    file_speeds: Mapping[str, float | None]
    elapsed: float

    def eta(self, snapshot: DownloadProgress) -> float | None:
        """Estimate the time left.

        Args:
            snapshot: The snapshot these stats belong to.

        Returns:
            The remaining bytes divided by the speed, or None without a speed.
        """
        if not self.speed:
            return None
        return (snapshot.total_bytes - snapshot.done_bytes) / self.speed

    def average_speed(self, snapshot: DownloadProgress) -> float | None:
        """Return the average speed of this run, leaving out resumed bytes.

        Args:
            snapshot: The snapshot these stats belong to.

        Returns:
            The bytes downloaded in this run per second, or None before any time passed.
        """
        if self.elapsed <= 0:
            return None
        return (snapshot.done_bytes - snapshot.started_bytes) / self.elapsed


class StatsTracker:
    """Turns a stream of snapshots into speeds and elapsed time."""

    def __init__(self) -> None:
        """Create the tracker."""
        self._started_at: float | None = None
        self._total = SpeedMeter()
        self._files: dict[str, SpeedMeter] = {}

    def add(self, snapshot: DownloadProgress) -> Stats:
        """Measure one snapshot.

        Args:
            snapshot: The newest snapshot.

        Returns:
            The speeds and elapsed time as of this snapshot.
        """
        now = snapshot.timestamp
        if self._started_at is None:
            self._started_at = now
        file_speeds: dict[str, float | None] = {}
        for file in snapshot.files:
            if file.state in MEASURED_STATES:
                meter = self._files.setdefault(file.path, SpeedMeter())
                file_speeds[file.path] = meter.add(now, file.bytes_on_disk)
            else:
                self._files.pop(file.path, None)
        return Stats(
            speed=self._total.add(now, snapshot.done_bytes),
            file_speeds=file_speeds,
            elapsed=now - self._started_at,
        )
