"""Download files in parallel with curl, with retries, checksums and progress reports."""

import logging
import threading
from collections.abc import Generator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .curl import Curl, CurlResult
from .errors import DownloadError, StoppedError
from .hub import RemoteFile
from .layout import Layout
from .progress import PROGRESS_EVENT, FileState, ProgressCallback, ProgressTracker
from .utils import file_size

logger = logging.getLogger(__name__)

MAX_FAILURES = 10  # failed attempts in a row that download nothing new
MAX_WAIT_SECONDS = 60
MAX_CHECKSUM_MISMATCHES = 2
PROGRESS_INTERVAL_SECONDS = 0.25


@dataclass
class _Backoff:
    """The retry policy for curl attempts of one file.

    The wait doubles after each attempt that downloads nothing new, and resets as soon as
    an attempt makes progress.

    Attributes:
        failures: Failed attempts in a row that downloaded nothing new.
    """

    failures: int = 0

    def record(self, *, progressed: bool) -> None:
        """Count a failed attempt.

        Args:
            progressed: Whether the partial file grew during the attempt.
        """
        self.failures = 0 if progressed else self.failures + 1

    @property
    def exhausted(self) -> bool:
        """Whether there were too many attempts in a row with no progress."""
        return self.failures > MAX_FAILURES

    @property
    def wait_seconds(self) -> int:
        """How long to wait before the next attempt."""
        return min(2**self.failures, MAX_WAIT_SECONDS)


class Downloader:
    """Downloads files into a layout, several at a time.

    On any error or Ctrl+C it stops every download and keeps the partial files, so the next
    run continues them.
    """

    def __init__(
        self,
        layout: Layout,
        curl: Curl,
        files: Sequence[RemoteFile],
        on_progress: ProgressCallback | None = None,
    ) -> None:
        """Create the downloader and measure the partial files already on disk.

        Files that share a partial path (identical blobs in the cache) are fetched once.

        Args:
            layout: Where the files go on disk.
            curl: Runs the curl processes.
            files: The files to download.
            on_progress: Called about every 0.25 s with a progress snapshot.
        """
        self._layout = layout
        self._curl = curl
        self._on_progress = on_progress
        self._stop = threading.Event()
        by_partial: dict[Path, RemoteFile] = {}
        for file in files:
            by_partial.setdefault(layout.partial_path(file), file)
        self._files = tuple(by_partial.values())
        self._tracker = ProgressTracker(self._files, self._partial_size)

    def download_all(self, max_workers: int) -> None:
        """Download the files in parallel until each one is complete and checked.

        Args:
            max_workers: How many files download at the same time.

        Raises:
            DownloadError: If a file cannot be downloaded.
            StoppedError: If the download was stopped.
        """
        with self._progress_reports():
            self._run_pool(max_workers)

    def _run_pool(self, max_workers: int) -> None:
        """Run one task per file in a thread pool, and stop all of them on the first error.

        Args:
            max_workers: The number of threads.

        Raises:
            DownloadError: Raised again from the first task that fails.
            StoppedError: Raised again if a task saw that the download was stopped.
        """
        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = [pool.submit(self._download_file, file) for file in self._files]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            self._cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _cancel(self) -> None:
        """Tell every task to stop and end the running curl processes."""
        self._stop.set()
        self._curl.terminate_all()

    @contextmanager
    def _progress_reports(self) -> Generator[None, None, None]:
        """Report progress at the start, every 0.25 s while the block runs, and at the end.

        Yields:
            Nothing. Progress is reported while the `with` block runs.
        """
        callback = self._on_progress
        if callback is None:
            yield
            return

        finished = threading.Event()

        def watch() -> None:
            """Report progress until the block ends."""
            while not finished.wait(PROGRESS_INTERVAL_SECONDS):
                callback(self._tracker.snapshot())

        callback(self._tracker.snapshot())
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            yield
        finally:
            finished.set()
            watcher.join()
            callback(self._tracker.snapshot())

    def _partial_size(self, file: RemoteFile) -> int:
        """Return the size of a file's partial file.

        Args:
            file: A file of this run.

        Returns:
            The size in bytes, or 0 if there is no partial file.
        """
        return file_size(self._layout.partial_path(file))

    def _download_file(self, file: RemoteFile) -> None:
        """Download one file until it is complete and its checksum matches.

        Args:
            file: The file to download.

        Raises:
            DownloadError: If the server refuses the file, the disk cannot be written, the
                checksum keeps failing, or too many attempts in a row make no progress.
            StoppedError: If the download was stopped.
        """
        partial = self._layout.partial_path(file)
        partial.parent.mkdir(parents=True, exist_ok=True)
        backoff = _Backoff()
        mismatches = 0
        while True:
            self._raise_if_stopped()
            partial.touch()  # an empty file needs no download, but its checksum is still checked
            have = self._usable_partial_size(file, partial)
            if have < file.size:
                self._fetch_once(file, partial, have, backoff)
            elif self._verify_and_finish(file, partial):
                return
            else:
                mismatches += 1
                if mismatches > MAX_CHECKSUM_MISMATCHES:
                    raise DownloadError(f"{file.path}: checksum keeps failing.")

    def _raise_if_stopped(self) -> None:
        """Stop the current task if the download was stopped.

        Raises:
            StoppedError: If the download was stopped.
        """
        if self._stop.is_set():
            raise StoppedError

    @staticmethod
    def _usable_partial_size(file: RemoteFile, partial: Path) -> int:
        """Return the size of the partial file, deleting it if it is bigger than the file.

        Args:
            file: The file being downloaded.
            partial: Its partial file.

        Returns:
            The number of bytes that can be kept.
        """
        have = file_size(partial)
        if have > file.size:
            partial.unlink()
            return 0
        return have

    def _verify_and_finish(self, file: RemoteFile, partial: Path) -> bool:
        """Check the checksum of a complete partial file and move it into place if it matches.

        A file that does not match is deleted, so the next attempt downloads it again.

        Args:
            file: The file being downloaded.
            partial: Its complete partial file.

        Returns:
            True if the file matched and is now in place, False if it was deleted.
        """
        self._tracker.set_state(file, FileState.VERIFYING)
        if file.is_lfs:
            logger.info("Checking the checksum of %s ...", file.path, extra=PROGRESS_EVENT)
        if not file.checksum_matches(partial):
            logger.warning(
                "%s: checksum does not match. Downloading it again from the start.", file.path
            )
            partial.unlink()
            return False
        self._layout.finish(file)
        self._tracker.set_state(file, FileState.DONE)
        logger.info("Done: %s", file.path, extra=PROGRESS_EVENT)
        return True

    def _fetch_once(self, file: RemoteFile, partial: Path, have: int, backoff: _Backoff) -> None:
        """Run curl once, and wait before the next attempt if it did not complete the file.

        Args:
            file: The file being downloaded.
            partial: Its partial file.
            have: The size of the partial file before this attempt.
            backoff: The retry policy of this file.

        Raises:
            DownloadError: If the error cannot be fixed by a retry, or if too many attempts in
                a row made no progress.
            StoppedError: If the download was stopped.
        """
        self._tracker.set_state(file, FileState.DOWNLOADING)
        result = self._curl.fetch(file.url, partial)
        self._raise_if_stopped()
        if result.succeeded and file_size(partial) == file.size:
            return  # the caller checks the checksum next
        self._handle_curl_error(file, partial, result)

        backoff.record(progressed=file_size(partial) > have)
        if backoff.exhausted:
            raise DownloadError(
                f"{file.path}: gave up after {backoff.failures} attempts with no progress "
                f"(curl exit code {result.exit_code}: {result.error}). "
                "Run the download again to continue."
            )
        wait = backoff.wait_seconds
        self._tracker.set_retrying(file, wait_seconds=wait, reason=result.reason)
        logger.warning(
            "%s: download interrupted (curl exit code %d). Continuing in %d s.",
            file.path,
            result.exit_code,
            wait,
            extra=PROGRESS_EVENT,
        )
        self._stop.wait(wait)

    @staticmethod
    def _handle_curl_error(file: RemoteFile, partial: Path, result: CurlResult) -> None:
        """Act on a curl run that did not complete the file.

        Fatal errors are raised. A server that cannot continue the file gets a fresh start.
        Every other error is left to the retry policy.

        Args:
            file: The file being downloaded.
            partial: Its partial file.
            result: The outcome of the curl run.

        Raises:
            DownloadError: If the server refused the file or the disk could not be written.
        """
        if result.is_fatal_http_error:
            raise DownloadError(
                f"{file.path}: the server answered HTTP {result.http_code}. {result.error}"
            )
        if result.is_write_error:
            raise DownloadError(
                f"{file.path}: could not write to disk (is it full?). {result.error}"
            )
        if result.is_range_error:
            logger.warning(
                "%s: the server cannot continue this file. Starting it again.", file.path
            )
            partial.unlink(missing_ok=True)
