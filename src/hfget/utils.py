"""Small helpers for file sizes and free disk space."""

import shutil
from pathlib import Path

from .errors import DownloadError


def file_size(path: Path) -> int:
    """Return the size of a file.

    Args:
        path: The file to measure.

    Returns:
        The size in bytes, or 0 if the file does not exist.
    """
    return path.stat().st_size if path.exists() else 0


def human_size(n: float) -> str:
    """Format a size the way `hf download --dry-run` does.

    Args:
        n: A size in bytes.

    Returns:
        The size with one decimal and a base-1000 unit, for example "5.3G".
    """
    for unit in ("", "K", "M", "G", "T"):
        if n < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}P"


def ensure_free_space(folder: Path, needed: int) -> None:
    """Check that the disk that holds a folder has enough free space.

    The folder does not need to exist yet: the check uses its nearest existing parent.

    Args:
        folder: Where the files will be written.
        needed: The number of bytes that will be written.

    Raises:
        DownloadError: If the disk has less free space than needed.
    """
    existing = folder
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < needed:
        raise DownloadError(
            f"Not enough disk space in {existing}: "
            f"need {human_size(needed)}, have {human_size(free)}."
        )
