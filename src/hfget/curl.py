"""Run curl to download a URL into a partial file, continuing where the file stopped."""

import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from .errors import DownloadError

CONNECT_TIMEOUT_SECONDS = 30
STALL_SECONDS = 60  # curl gives up if no data arrives for this long
FATAL_HTTP_CODES = frozenset({401, 403, 404})

CURL_OK = 0
CURL_HTTP_ERROR = 22
CURL_WRITE_ERROR = 23
CURL_RANGE_ERROR = 33

EXIT_CODE_REASONS = {
    6: "could not resolve host",
    7: "could not connect",
    18: "transfer closed early",
    23: "could not write to disk",
    28: f"no data for {STALL_SECONDS} s",
    33: "server cannot resume",
    35: "TLS error",
    52: "empty reply",
    56: "connection reset",
}


@dataclass(frozen=True)
class CurlResult:
    """The outcome of one curl run.

    Attributes:
        exit_code: curl's exit code. It is negative when a signal stopped curl.
        http_code: The last HTTP status curl received, as text (for example "206").
        error: What curl wrote on stderr.
    """

    exit_code: int
    http_code: str
    error: str

    @property
    def succeeded(self) -> bool:
        """Whether curl finished without an error."""
        return self.exit_code == CURL_OK

    @property
    def is_fatal_http_error(self) -> bool:
        """Whether the server refused the file (HTTP 401, 403 or 404), so a retry cannot help."""
        return (
            self.exit_code == CURL_HTTP_ERROR
            and self.http_code.isdigit()
            and int(self.http_code) in FATAL_HTTP_CODES
        )

    @property
    def is_write_error(self) -> bool:
        """Whether curl could not write to the disk."""
        return self.exit_code == CURL_WRITE_ERROR

    @property
    def is_range_error(self) -> bool:
        """Whether the server cannot continue the partial file from its current size."""
        return self.exit_code == CURL_RANGE_ERROR

    @property
    def reason(self) -> str:
        """A short human reason for a failed run, for example "connection reset"."""
        if self.exit_code < 0:
            return "curl was stopped"
        if self.exit_code == CURL_HTTP_ERROR and self.http_code.isdigit():
            return f"HTTP {self.http_code}"
        return EXIT_CODE_REASONS.get(self.exit_code, f"curl exit code {self.exit_code}")


class Curl:
    """Starts curl downloads and can stop all running ones at once.

    The access token goes to curl on stdin, so it never shows in the process list.
    """

    def __init__(self, token: str | None) -> None:
        """Create the runner.

        Args:
            token: A Hugging Face access token, or None for anonymous downloads.

        Raises:
            DownloadError: If curl is not installed.
        """
        if shutil.which("curl") is None:
            raise DownloadError(
                "curl is not installed or not on PATH. Install it first, for example "
                "`sudo apt install curl` (Debian/Ubuntu) or `brew install curl` (macOS)."
            )
        self._token = token
        self._running: set[subprocess.Popen[str]] = set()
        self._stopped = False
        self._lock = threading.Lock()

    def fetch(self, url: str, output: Path) -> CurlResult:
        """Download a URL into a file, continuing from the end of the file if it exists.

        Args:
            url: The file URL.
            output: The partial file to write to.

        Returns:
            The exit code, the last HTTP status and the error text of the curl run.
        """
        proc = subprocess.Popen(
            self._command(url, output),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._lock:
            self._running.add(proc)
            if self._stopped:  # `terminate_all` ran while this process was starting
                proc.terminate()
        try:
            out, err = proc.communicate(self._config())
        finally:
            with self._lock:
                self._running.discard(proc)
        return CurlResult(proc.returncode, out.strip(), err.strip())

    def terminate_all(self) -> None:
        """Stop every running curl process, and any process started after this call."""
        with self._lock:
            self._stopped = True
            for proc in self._running:
                proc.terminate()

    def _command(self, url: str, output: Path) -> list[str]:
        """Build the curl command line. It never contains the token.

        Args:
            url: The file URL.
            output: The partial file to write to.

        Returns:
            The command and its arguments.
        """
        cmd = [
            "curl",
            "--location",  # follow the redirect to the file storage
            "--fail",
            "--silent",
            "--show-error",
            "--continue-at", "-",  # continue from the end of the partial file
            "--output", str(output),
            "--connect-timeout", str(CONNECT_TIMEOUT_SECONDS),
            "--speed-limit", "1",
            "--speed-time", str(STALL_SECONDS),
            "--write-out", "%{http_code}",
        ]  # fmt: skip
        if self._token:
            cmd += ["--config", "-"]  # read the auth header from stdin
        cmd.append(url)
        return cmd

    def _config(self) -> str:
        """Build the curl config that is sent on stdin.

        Returns:
            The `Authorization` header line if there is a token, else an empty string.
        """
        if not self._token:
            return ""
        return f'header = "Authorization: Bearer {self._token}"\n'
