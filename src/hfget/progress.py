"""Progress snapshots of a running download, and the tracker that builds them.

The worker threads record each file's state in a `ProgressTracker`. A polling thread asks
the tracker for an immutable `DownloadProgress` snapshot a few times per second and hands
it to the caller's `ProgressCallback`. The library never draws anything itself.
"""

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias

from .hub import RemoteFile

PROGRESS_EVENT_ATTRIBUTE = "progress_event"
PROGRESS_EVENT: Mapping[str, object] = MappingProxyType({PROGRESS_EVENT_ATTRIBUTE: True})
"""Pass as `extra` to mark a log message that a progress display already shows."""


class FileState(Enum):
    """Where a file is in its download."""

    QUEUED = "queued"
    DOWNLOADING = "downloading"
    RETRYING = "retrying"
    VERIFYING = "verifying"
    DONE = "done"


@dataclass(frozen=True)
class RetryInfo:
    """Why a file waits, and until when.

    Attributes:
        attempt: The number of the coming retry for this file, starting at 1.
        next_attempt_at: When the next attempt starts, on the `time.monotonic()` clock.
        reason: A short human reason, for example "connection reset".
    """

    attempt: int
    next_attempt_at: float
    reason: str


@dataclass(frozen=True)
class FileProgress:
    """The progress of one file.

    Attributes:
        path: The path inside the repo.
        size: The size in bytes.
        bytes_on_disk: The bytes downloaded so far, including bytes from earlier runs.
        state: Where the file is in its download.
        retry: Why and until when the file waits. Set only in the `RETRYING` state.
    """

    path: str
    size: int
    bytes_on_disk: int
    state: FileState
    retry: RetryInfo | None = None


@dataclass(frozen=True)
class DownloadProgress:
    """A snapshot of the whole download.

    Attributes:
        files: The progress of every file this run downloads, in repo order.
        started_bytes: The bytes that were already on disk when this run began. Speeds
            should leave them out.
        done_bytes: The bytes on disk now.
        total_bytes: The total size of the files.
        retries: How many retries this run made so far, over all files.
        timestamp: When the snapshot was taken, on the `time.monotonic()` clock.
    """

    files: tuple[FileProgress, ...]
    started_bytes: int
    done_bytes: int
    total_bytes: int
    retries: int
    timestamp: float


ProgressCallback: TypeAlias = Callable[[DownloadProgress], None]
"""Receives a progress snapshot about every 0.25 s while files download."""


def is_progress_event(record: logging.LogRecord) -> bool:
    """Tell whether a log record repeats what a progress display already shows.

    These are the "Downloading N of M files", "Checking the checksum", "Done" and
    "download interrupted" messages. A program that draws its own progress can drop them.

    Args:
        record: A log record from an `hfget` logger.

    Returns:
        True if the record was logged with `extra=PROGRESS_EVENT`.
    """
    return bool(getattr(record, PROGRESS_EVENT_ATTRIBUTE, False))


@dataclass
class _Entry:
    """The mutable state of one file inside the tracker.

    Attributes:
        file: The file.
        initial_bytes: The bytes on disk when the tracker was created.
        state: Where the file is in its download.
        retries: How many retries this file made so far.
        retry: The pending retry, while the file is in the `RETRYING` state.
    """

    file: RemoteFile
    initial_bytes: int
    state: FileState = FileState.QUEUED
    retries: int = 0
    retry: RetryInfo | None = None


class ProgressTracker:
    """A thread-safe record of each file's state that builds progress snapshots."""

    def __init__(
        self, files: Sequence[RemoteFile], partial_size: Callable[[RemoteFile], int]
    ) -> None:
        """Create the tracker and measure what is already on disk.

        Args:
            files: The files of this run. Their paths must be unique.
            partial_size: Returns the current size of a file's partial file.
        """
        self._partial_size = partial_size
        self._lock = threading.Lock()
        self._entries = {
            file.path: _Entry(file, min(partial_size(file), file.size)) for file in files
        }
        self._started_bytes = sum(entry.initial_bytes for entry in self._entries.values())

    def set_state(self, file: RemoteFile, state: FileState) -> None:
        """Record a new state for a file. This clears any pending retry.

        Args:
            file: A file of this run.
            state: Its new state.
        """
        with self._lock:
            entry = self._entries[file.path]
            entry.state = state
            entry.retry = None

    def set_retrying(self, file: RemoteFile, *, wait_seconds: float, reason: str) -> None:
        """Record that a file waits before its next attempt.

        Args:
            file: A file of this run.
            wait_seconds: How long the file waits.
            reason: A short human reason for the failed attempt.
        """
        with self._lock:
            entry = self._entries[file.path]
            entry.retries += 1
            entry.state = FileState.RETRYING
            entry.retry = RetryInfo(entry.retries, time.monotonic() + wait_seconds, reason)

    def snapshot(self) -> DownloadProgress:
        """Build an immutable snapshot of the download.

        The partial files are measured outside the lock, so workers are never blocked by
        disk access.

        Returns:
            The current progress.
        """
        with self._lock:
            entries = [
                (entry.file, entry.state, entry.retry, entry.initial_bytes)
                for entry in self._entries.values()
            ]
            retries = sum(entry.retries for entry in self._entries.values())
        files = tuple(
            FileProgress(
                path=file.path,
                size=file.size,
                bytes_on_disk=self._bytes_on_disk(file, state, initial_bytes),
                state=state,
                retry=retry,
            )
            for file, state, retry, initial_bytes in entries
        )
        return DownloadProgress(
            files=files,
            started_bytes=self._started_bytes,
            done_bytes=sum(file.bytes_on_disk for file in files),
            total_bytes=sum(file.size for file in files),
            retries=retries,
            timestamp=time.monotonic(),
        )

    def _bytes_on_disk(self, file: RemoteFile, state: FileState, initial_bytes: int) -> int:
        """Return the downloaded bytes of a file in a given state.

        A file being checked or done counts as complete, so the count does not drop while
        its partial file is moved into place. A queued file has not changed since the start.

        Args:
            file: The file.
            state: Its state.
            initial_bytes: Its bytes on disk when the tracker was created.

        Returns:
            The bytes on disk, between 0 and the file size.
        """
        if state in (FileState.VERIFYING, FileState.DONE):
            return file.size
        if state is FileState.QUEUED:
            return initial_bytes
        return min(self._partial_size(file), file.size)
