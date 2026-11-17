"""Where downloaded files go on disk.

Two layouts are supported, both the same as `hf download` writes them:

- `LocalDirLayout`: the files go into a folder you choose (`--local-dir`).
- `CacheLayout`: the files go into the Hugging Face cache (`blobs/`, `snapshots/`, `refs/`).
"""

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from huggingface_hub import constants
from huggingface_hub.file_download import repo_folder_name

from .hub import RemoteFile
from .utils import file_size


class Layout(ABC):
    """How files are placed on disk while they download and once they are finished.

    Attributes:
        root: The folder where the files appear under their repo paths.
        commit: The commit the files come from.
    """

    def __init__(self, root: Path, commit: str) -> None:
        """Create the layout.

        Args:
            root: The folder where the files appear under their repo paths.
            commit: The commit the files come from.
        """
        self.root = root
        self.commit = commit

    @abstractmethod
    def final_path(self, file: RemoteFile) -> Path:
        """Return where the finished file is stored.

        Args:
            file: A repo file.

        Returns:
            The path of the finished file.
        """

    @abstractmethod
    def partial_path(self, file: RemoteFile) -> Path:
        """Return where the file is written while it downloads.

        Args:
            file: A repo file.

        Returns:
            The path of the partial (`.incomplete`) file.
        """

    @abstractmethod
    def prepare(self) -> None:
        """Create what the layout needs before any file is checked or downloaded."""

    @abstractmethod
    def finalize(self, files: Iterable[RemoteFile]) -> None:
        """Do the work that is needed once all files are on disk.

        Args:
            files: Every selected file, including the ones that were already done.
        """

    def is_done(self, file: RemoteFile) -> bool:
        """Tell whether a file is already downloaded and matches the repo.

        Args:
            file: A repo file.

        Returns:
            True if the finished file exists with the right size.
        """
        final = self.final_path(file)
        return final.exists() and file_size(final) == file.size

    def finish(self, file: RemoteFile) -> None:
        """Move a complete, checked partial file to its final place.

        Args:
            file: The repo file whose partial file is complete.
        """
        final = self.final_path(file)
        final.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path(file).replace(final)


class LocalDirLayout(Layout):
    """Files go to `<local_dir>/<path>`, with the bookkeeping of `hf download --local-dir`.

    The partial file and the record of each file live in
    `<local_dir>/.cache/huggingface/download/`.
    """

    def __init__(self, local_dir: Path, commit: str) -> None:
        """Create the layout.

        Args:
            local_dir: The folder that receives the files.
            commit: The commit the files come from.
        """
        super().__init__(root=local_dir, commit=commit)
        self._bookkeeping = local_dir / ".cache" / "huggingface"
        self._downloads = self._bookkeeping / "download"

    def final_path(self, file: RemoteFile) -> Path:
        """Return `<local_dir>/<path>`.

        Args:
            file: A repo file.

        Returns:
            The path of the finished file.
        """
        return self.root / file.path

    def partial_path(self, file: RemoteFile) -> Path:
        """Return `<local_dir>/.cache/huggingface/download/<path>.incomplete`.

        Args:
            file: A repo file.

        Returns:
            The path of the partial file.
        """
        return self._downloads / f"{file.path}.incomplete"

    def prepare(self) -> None:
        """Create the bookkeeping folder with a `.gitignore`, like `hf download` does."""
        self._bookkeeping.mkdir(parents=True, exist_ok=True)
        (self._bookkeeping / ".gitignore").write_text("*\n")

    def finalize(self, files: Iterable[RemoteFile]) -> None:
        """Do nothing: `finish` already wrote the record of each file.

        Args:
            files: Every selected file (not used).
        """

    def is_done(self, file: RemoteFile) -> bool:
        """Tell whether a file is downloaded: right size, and its record has the same etag.

        Args:
            file: A repo file.

        Returns:
            True if the file can be skipped.
        """
        metadata = self._metadata_path(file)
        if not super().is_done(file) or not metadata.exists():
            return False
        lines = metadata.read_text().splitlines()
        return len(lines) >= 2 and lines[1] == file.etag

    def finish(self, file: RemoteFile) -> None:
        """Move the file into place and write its record, the same one `hf download` writes.

        Args:
            file: The repo file whose partial file is complete.
        """
        super().finish(file)
        metadata = self._metadata_path(file)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(f"{self.commit}\n{file.etag}\n{time.time()}\n")

    def _metadata_path(self, file: RemoteFile) -> Path:
        """Return the path of a file's record.

        Args:
            file: A repo file.

        Returns:
            `<local_dir>/.cache/huggingface/download/<path>.metadata`.
        """
        return self._downloads / f"{file.path}.metadata"


class CacheLayout(Layout):
    """Files go into the Hugging Face cache, so the `huggingface_hub` library finds them.

    A finished file is a blob named by its etag. `snapshots/<commit>/<path>` is a relative
    symlink to the blob, and `refs/<revision>` holds the commit of a branch or tag.
    """

    def __init__(self, storage: Path, commit: str, revision: str | None) -> None:
        """Create the layout.

        Args:
            storage: The repo folder inside the cache (see `repo_cache_folder`).
            commit: The commit the files come from.
            revision: The branch, tag or commit the user asked for. None means "main".
        """
        super().__init__(root=storage / "snapshots" / commit, commit=commit)
        self._storage = storage
        self._revision = revision or "main"

    def final_path(self, file: RemoteFile) -> Path:
        """Return `<storage>/blobs/<etag>`.

        Args:
            file: A repo file.

        Returns:
            The path of the blob.
        """
        return self._storage / "blobs" / file.etag

    def partial_path(self, file: RemoteFile) -> Path:
        """Return `<storage>/blobs/<etag>.incomplete`.

        Args:
            file: A repo file.

        Returns:
            The path of the partial blob.
        """
        return self._storage / "blobs" / f"{file.etag}.incomplete"

    def prepare(self) -> None:
        """Do nothing: the cache folders are created when files are written."""

    def finalize(self, files: Iterable[RemoteFile]) -> None:
        """Link every file into the snapshot and record the commit of the revision.

        `refs/<revision>` is written only when the revision is a branch or tag name.

        Args:
            files: Every selected file, including the ones that were already done.
        """
        for file in files:
            self._link_pointer(file)
        if self._revision != self.commit:
            ref = self._storage / "refs" / self._revision
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(self.commit)

    def finish(self, file: RemoteFile) -> None:
        """Move the blob into place and link it into the snapshot.

        Args:
            file: The repo file whose partial file is complete.
        """
        super().finish(file)
        self._link_pointer(file)

    def _link_pointer(self, file: RemoteFile) -> None:
        """Point `snapshots/<commit>/<path>` at its blob with a relative symlink.

        Args:
            file: A repo file whose blob is on disk.
        """
        pointer = self.root / file.path
        pointer.parent.mkdir(parents=True, exist_ok=True)
        if pointer.is_symlink() or pointer.exists():
            pointer.unlink()
        pointer.symlink_to(os.path.relpath(self.final_path(file), pointer.parent))


def repo_cache_folder(repo_id: str, repo_type: str, cache_dir: str | Path | None) -> Path:
    """Return the folder of a repo inside the Hugging Face cache.

    Args:
        repo_id: The repo, for example "Qwen/Qwen3-0.6B".
        repo_type: The repo type.
        cache_dir: The cache folder, or None for the default `huggingface_hub` cache.

    Returns:
        For example `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B`.
    """
    cache = Path(cache_dir) if cache_dir else Path(constants.HF_HUB_CACHE)
    return cache / repo_folder_name(repo_id=repo_id, repo_type=repo_type)
