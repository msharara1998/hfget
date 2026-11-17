"""Exceptions raised by the downloader."""


class DownloadError(Exception):
    """A problem that retrying will not fix."""


class StoppedError(Exception):
    """The download was stopped before it finished."""
