"""The public functions: plan a download, run a plan, or do both in one call.

The functions report what they do through the `logging` module (logger names start with
`hfget`) and raise exceptions instead of exiting.

Example:
    >>> from hfget import download
    >>> download("Qwen/Qwen3-0.6B", filenames=["config.json"], local_dir="models/qwen")
    PosixPath('models/qwen/config.json')
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import get_token

from .curl import Curl
from .downloader import Downloader
from .hub import RemoteFile, list_repo_files, select_files, split_repo_id
from .layout import CacheLayout, Layout, LocalDirLayout, repo_cache_folder
from .progress import PROGRESS_EVENT, ProgressCallback
from .utils import ensure_free_space, file_size

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadPlan:
    """What a download will do. `--dry-run` shows this.

    Attributes:
        repo_id: The repo, without a type prefix.
        repo_type: "model", "dataset", "space" or "kernel".
        commit: The commit the files come from.
        files: Every selected file.
        pending: The selected files that still need to be downloaded.
        bytes_to_download: The bytes still missing for the pending files. Bytes already in
            partial files are not counted.
        path: What `download` returns: the requested file if exactly one file was asked
            for by name, otherwise the folder.
        layout: Where the files go on disk.
    """

    repo_id: str
    repo_type: str
    commit: str
    files: tuple[RemoteFile, ...]
    pending: tuple[RemoteFile, ...]
    bytes_to_download: int
    path: Path
    layout: Layout


def plan_download(  # noqa: PLR0913 - the same keyword options as `hf download`
    repo_id: str,
    *,
    filenames: Sequence[str] | None = None,
    repo_type: str = "model",
    revision: str | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    force_download: bool = False,
    token: str | None = None,
) -> DownloadPlan:
    """List the repo, select the files and find the ones that still need to be downloaded.

    Nothing is downloaded. With `local_dir`, the bookkeeping folder
    `<local_dir>/.cache/huggingface` is created, as `hf download` does.

    Args:
        repo_id: The repo, for example "Qwen/Qwen3-0.6B". The forms `datasets/org/name`,
            `spaces/org/name` and `hf://...` also work.
        filenames: Only these files. A name that ends with "/" means a whole folder.
        repo_type: "model", "dataset", "space" or "kernel". A prefix in `repo_id` wins.
        revision: A branch, tag or commit. None means the default branch.
        include: Glob patterns of files to download. Ignored when `filenames` is set.
        exclude: Glob patterns of files to skip. Ignored when `filenames` is set.
        cache_dir: The Hugging Face cache folder, used when `local_dir` is not set.
            None means the default cache.
        local_dir: Save the files in this folder instead of the cache.
        force_download: Download files again even if they are already complete. Partial
            files are still continued.
        token: A Hugging Face access token. None means the saved token, if any.

    Returns:
        The plan.

    Raises:
        DownloadError: If the repo cannot be listed or a requested file is not in it.
    """
    repo_id, repo_type = split_repo_id(repo_id, repo_type)
    commit, files = list_repo_files(repo_id, repo_type, revision, token)
    names = list(filenames or [])
    selected = select_files(files, names, include, exclude)

    layout: Layout
    if local_dir:
        layout = LocalDirLayout(Path(local_dir), commit)
    else:
        layout = CacheLayout(repo_cache_folder(repo_id, repo_type, cache_dir), commit, revision)
    layout.prepare()

    pending = [file for file in selected if force_download or not layout.is_done(file)]
    return DownloadPlan(
        repo_id=repo_id,
        repo_type=repo_type,
        commit=commit,
        files=tuple(selected),
        pending=tuple(pending),
        bytes_to_download=sum(file.size - file_size(layout.partial_path(file)) for file in pending),
        path=_result_path(layout.root, names, selected),
        layout=layout,
    )


def execute_plan(
    plan: DownloadPlan,
    *,
    token: str | None = None,
    max_workers: int = 8,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download the pending files of a plan and complete the disk layout.

    Args:
        plan: The plan from `plan_download`.
        token: A Hugging Face access token. None means the saved token, if any.
        max_workers: How many files download at the same time.
        on_progress: Called about every 0.25 s with a `DownloadProgress` snapshot of the
            pending files.

    Returns:
        The downloaded file if exactly one file was asked for by name, otherwise the folder.

    Raises:
        DownloadError: If there is not enough disk space or a file cannot be downloaded.
            Partial files are kept, so a new call continues them.
        StoppedError: If the download was stopped.
    """
    if plan.pending:
        ensure_free_space(plan.layout.root, plan.bytes_to_download)
        logger.info(
            "Downloading %d of %d files from %s (%s).",
            len(plan.pending),
            len(plan.files),
            plan.repo_id,
            plan.commit[:8],
            extra=PROGRESS_EVENT,
        )
        curl = Curl(token or get_token())
        Downloader(plan.layout, curl, plan.pending, on_progress).download_all(max_workers)
    else:
        logger.info("All files are already downloaded.")
    plan.layout.finalize(plan.files)
    return plan.path


def download(  # noqa: PLR0913 - the same keyword options as `hf download`
    repo_id: str,
    *,
    filenames: Sequence[str] | None = None,
    repo_type: str = "model",
    revision: str | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    force_download: bool = False,
    token: str | None = None,
    max_workers: int = 8,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download files from a Hub repo, continuing partial files from where they stopped.

    This is `plan_download` followed by `execute_plan`.

    Args:
        repo_id: The repo, for example "Qwen/Qwen3-0.6B". The forms `datasets/org/name`,
            `spaces/org/name` and `hf://...` also work.
        filenames: Only these files. A name that ends with "/" means a whole folder.
        repo_type: "model", "dataset", "space" or "kernel". A prefix in `repo_id` wins.
        revision: A branch, tag or commit. None means the default branch.
        include: Glob patterns of files to download. Ignored when `filenames` is set.
        exclude: Glob patterns of files to skip. Ignored when `filenames` is set.
        cache_dir: The Hugging Face cache folder, used when `local_dir` is not set.
            None means the default cache.
        local_dir: Save the files in this folder instead of the cache.
        force_download: Download files again even if they are already complete. Partial
            files are still continued.
        token: A Hugging Face access token. None means the saved token, if any.
        max_workers: How many files download at the same time.
        on_progress: Called about every 0.25 s with a `DownloadProgress` snapshot of the
            pending files.

    Returns:
        The downloaded file if exactly one file was asked for by name, otherwise the folder.

    Raises:
        DownloadError: If a problem cannot be fixed by retrying. Partial files are kept.
        StoppedError: If the download was stopped.
    """
    plan = plan_download(
        repo_id,
        filenames=filenames,
        repo_type=repo_type,
        revision=revision,
        include=include,
        exclude=exclude,
        cache_dir=cache_dir,
        local_dir=local_dir,
        force_download=force_download,
        token=token,
    )
    return execute_plan(plan, token=token, max_workers=max_workers, on_progress=on_progress)


def _result_path(root: Path, filenames: Sequence[str], selected: Sequence[RemoteFile]) -> Path:
    """Choose the path a download returns.

    Args:
        root: The folder where the files appear under their repo paths.
        filenames: The file names the user asked for.
        selected: The selected files.

    Returns:
        The file if one exact file name was asked for and it matched one file, else `root`.
    """
    single_file = len(filenames) == 1 and len(selected) == 1 and not filenames[0].endswith("/")
    return root / selected[0].path if single_file else root
