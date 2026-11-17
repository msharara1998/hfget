"""hfget: download Hugging Face Hub files with curl, continuing partial files.

`download` fetches files from a Hub repo with curl and continues partial files from where
they stopped. `plan_download` shows
what a download would do without downloading, and `execute_plan` runs such a plan.

The functions log messages through the `logging` module, under the `hfget`
logger, send `DownloadProgress` snapshots to an optional `ProgressCallback`, and raise
`DownloadError` or `StoppedError` when they cannot finish. They never print.
"""

import logging
from importlib.metadata import PackageNotFoundError, version

from .api import DownloadPlan, download, execute_plan, plan_download
from .errors import DownloadError, StoppedError
from .hub import RemoteFile
from .layout import Layout
from .progress import (
    DownloadProgress,
    FileProgress,
    FileState,
    ProgressCallback,
    RetryInfo,
    is_progress_event,
)

try:
    __version__ = version("hfget")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "DownloadError",
    "DownloadPlan",
    "DownloadProgress",
    "FileProgress",
    "FileState",
    "Layout",
    "ProgressCallback",
    "RemoteFile",
    "RetryInfo",
    "StoppedError",
    "__version__",
    "download",
    "execute_plan",
    "is_progress_event",
    "plan_download",
]

# A library adds no output handler of its own: the application decides where logs go.
logging.getLogger(__name__).addHandler(logging.NullHandler())
