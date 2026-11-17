"""Find the files of a Hugging Face repo and pick the ones to download.

The `huggingface_hub` library does the Hub work: it gets the commit, lists the files with
their sizes and checksums, matches the file name patterns and builds the download URLs.
"""

import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import filter_repo_objects

from .errors import DownloadError

logger = logging.getLogger(__name__)

REPO_TYPES = ("model", "dataset", "space", "kernel")
HUB_ATTEMPTS = 5  # attempts to reach the Hub before giving up
READ_CHUNK = 8 * 1024 * 1024  # bytes read at a time when computing a checksum
GLOB_CHARACTERS = "*?["


@dataclass(frozen=True)
class RemoteFile:
    """One file in a Hub repo, pinned to a commit.

    Attributes:
        path: The path inside the repo, for example "config.json".
        size: The size in bytes.
        etag: The checksum from the Hub: sha256 for LFS files, the git blob id for the others.
        is_lfs: Whether the file is stored with Git LFS.
        url: The download URL for this commit.
    """

    path: str
    size: int
    etag: str
    is_lfs: bool
    url: str

    def checksum_matches(self, local_path: Path) -> bool:
        """Compare a local file with the checksum from the Hub.

        Args:
            local_path: The downloaded file to check.

        Returns:
            True if the file's sha256 (LFS files) or git blob id (other files) equals the etag.
        """
        digest = hashlib.sha256() if self.is_lfs else hashlib.sha1(f"blob {self.size}\0".encode())
        with local_path.open("rb") as fh:
            while chunk := fh.read(READ_CHUNK):
                digest.update(chunk)
        return digest.hexdigest() == self.etag


def split_repo_id(raw: str, repo_type: str) -> tuple[str, str]:
    """Split a repo id that may carry its type, like `hf download` accepts.

    Accepted forms: `org/name`, `datasets/org/name` (also `models/`, `spaces/` and
    `kernels/`), each with an optional `hf://` prefix.

    Args:
        raw: The repo id as the user typed it.
        repo_type: The type to use when `raw` has no type prefix.

    Returns:
        The repo id without prefixes, and the repo type.
    """
    repo_id = raw.removeprefix("hf://")
    for kind in REPO_TYPES:
        prefix = f"{kind}s/"
        if repo_id.startswith(prefix):
            return repo_id.removeprefix(prefix), kind
    return repo_id, repo_type


def list_repo_files(
    repo_id: str, repo_type: str, revision: str | None, token: str | None
) -> tuple[str, list[RemoteFile]]:
    """List every file in a repo at the commit of a revision.

    Args:
        repo_id: The repo, for example "Qwen/Qwen3-0.6B".
        repo_type: "model", "dataset", "space" or "kernel".
        revision: A branch, tag or commit. None means the default branch.
        token: A Hugging Face access token. None means the saved token, if any.

    Returns:
        The commit hash, and the repo files with their sizes, checksums and URLs.

    Raises:
        DownloadError: If the repo is gated, not found or has no such revision, or if the
            Hub cannot be reached after several attempts.
    """
    commit, entries = _list_tree_with_retries(repo_id, repo_type, revision, token)
    return commit, [_to_remote_file(entry, repo_id, repo_type, commit) for entry in entries]


def _list_tree_with_retries(
    repo_id: str, repo_type: str, revision: str | None, token: str | None
) -> tuple[str, list[RepoFile]]:
    """Ask the Hub for the commit and the file tree, retrying network errors.

    Errors that a retry cannot fix are turned into a `DownloadError` at once.

    Args:
        repo_id: The repo.
        repo_type: The repo type.
        revision: A branch, tag or commit, or None for the default branch.
        token: A Hugging Face access token, or None.

    Returns:
        The commit hash and the file entries of the tree.

    Raises:
        DownloadError: If the repo is gated, not found or has no such revision, or if the
            Hub cannot be reached after `HUB_ATTEMPTS` attempts.
    """
    attempt = 1
    while True:
        try:
            return _list_tree(repo_id, repo_type, revision, token)
        except DownloadError:
            raise
        # GatedRepoError is a subclass of RepositoryNotFoundError, so it must come first.
        except GatedRepoError as e:
            raise DownloadError(
                f"{repo_id} is gated. Accept its terms on huggingface.co, then pass a token "
                "(`token=` in Python, `--token` on the command line) or run `hf auth login`."
            ) from e
        except RepositoryNotFoundError as e:
            raise DownloadError(
                f"Repo not found: {repo_id} (type: {repo_type}). If it is private, pass a token "
                "(`token=` in Python, `--token` on the command line)."
            ) from e
        except RevisionNotFoundError as e:
            raise DownloadError(f"Revision not found in {repo_id}: {revision}") from e
        except Exception as e:  # network errors: try again
            if attempt == HUB_ATTEMPTS:
                raise DownloadError(f"Could not reach the Hugging Face Hub: {e}") from e
            wait = 2**attempt
            logger.warning("Could not reach the Hub (%s). Trying again in %d s.", e, wait)
            time.sleep(wait)
        attempt += 1


def _list_tree(
    repo_id: str, repo_type: str, revision: str | None, token: str | None
) -> tuple[str, list[RepoFile]]:
    """Ask the Hub once for the commit of a revision and the files at that commit.

    `list_repo_tree` is used instead of `repo_info(files_metadata=True).siblings`, because
    kernel repos have no siblings.

    Args:
        repo_id: The repo.
        repo_type: The repo type.
        revision: A branch, tag or commit, or None for the default branch.
        token: A Hugging Face access token, or None.

    Returns:
        The commit hash and the file entries of the tree (folders are left out).

    Raises:
        DownloadError: If the Hub does not return a commit.
    """
    api = HfApi()
    commit = api.repo_info(repo_id, repo_type=repo_type, revision=revision, token=token).sha
    if commit is None:
        raise DownloadError(f"The Hub did not return a commit for {repo_id}.")
    tree = api.list_repo_tree(
        repo_id, recursive=True, revision=commit, repo_type=repo_type, token=token
    )
    return commit, [entry for entry in tree if isinstance(entry, RepoFile)]


def _to_remote_file(entry: RepoFile, repo_id: str, repo_type: str, commit: str) -> RemoteFile:
    """Turn a tree entry from the Hub into a `RemoteFile`.

    Args:
        entry: The file entry from `HfApi.list_repo_tree`.
        repo_id: The repo.
        repo_type: The repo type.
        commit: The commit the URL is pinned to.

    Returns:
        The file with its checksum and download URL.
    """
    lfs = entry.lfs
    return RemoteFile(
        path=entry.path,
        size=entry.size,
        etag=lfs.sha256 if lfs is not None else entry.blob_id,
        is_lfs=lfs is not None,
        url=hf_hub_url(repo_id, entry.path, repo_type=repo_type, revision=commit),
    )


def select_files(
    files: Sequence[RemoteFile],
    filenames: Sequence[str],
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
) -> list[RemoteFile]:
    """Pick the files to download the same way `hf download` does.

    Explicit file names win over the `include` and `exclude` patterns.

    Args:
        files: Every file in the repo.
        filenames: File names or patterns the user asked for. A name that ends with "/"
            means a whole folder. Empty means "use the patterns".
        include: Glob patterns of files to keep, or None to keep all.
        exclude: Glob patterns of files to leave out, or None.

    Returns:
        The selected files, in repo order.

    Raises:
        DownloadError: If an exact file name (not a folder or a pattern) is not in the repo.
    """
    if not filenames:
        return _filter(files, include, exclude)

    if include or exclude:
        logger.warning("Ignoring --include and --exclude since filenames have been explicitly set.")
    names = {file.path for file in files}
    missing = [name for name in filenames if _is_exact_name(name) and name not in names]
    if missing:
        raise DownloadError(f"Not found in the repo: {', '.join(missing)}")
    return _filter(files, filenames, None)


def _filter(
    files: Sequence[RemoteFile], allow: Sequence[str] | None, ignore: Sequence[str] | None
) -> list[RemoteFile]:
    """Filter files by path with the `huggingface_hub` pattern matcher.

    Args:
        files: The files to filter.
        allow: Patterns a path must match, or None to allow all.
        ignore: Patterns a path must not match, or None.

    Returns:
        The files that pass both filters.
    """
    return list(
        filter_repo_objects(
            files,
            allow_patterns=list(allow) if allow is not None else None,
            ignore_patterns=list(ignore) if ignore is not None else None,
            key=_repo_path,
        )
    )


def _repo_path(file: RemoteFile) -> str:
    """Return the path of a file inside the repo.

    Args:
        file: A repo file.

    Returns:
        The path, used as the key for pattern matching.
    """
    return file.path


def _is_exact_name(name: str) -> bool:
    """Tell whether a requested name means one exact file.

    Args:
        name: A name from the user's file list.

    Returns:
        False for a folder (ends with "/") or a glob pattern, True otherwise.
    """
    return not name.endswith("/") and not any(c in name for c in GLOB_CHARACTERS)
