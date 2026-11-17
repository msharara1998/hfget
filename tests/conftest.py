"""Shared helpers: repo files with real checksums, and the two disk layouts."""

import hashlib

import pytest

from hfget.hub import RemoteFile
from hfget.layout import CacheLayout, LocalDirLayout

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def git_blob_sha1(content: bytes) -> str:
    """The etag of a non-LFS file: the git blob id."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def make_remote_file(
    path: str, content: bytes = b"", *, lfs: bool = False, url: str = "https://hub.test/file"
) -> RemoteFile:
    """A repo file whose size and etag match `content`."""
    etag = hashlib.sha256(content).hexdigest() if lfs else git_blob_sha1(content)
    return RemoteFile(path=path, size=len(content), etag=etag, is_lfs=lfs, url=url)


@pytest.fixture
def remote_file():
    return make_remote_file


@pytest.fixture
def local_layout(tmp_path):
    return LocalDirLayout(tmp_path / "local", COMMIT)


@pytest.fixture
def cache_layout(tmp_path):
    return CacheLayout(tmp_path / "cache" / "models--org--name", COMMIT, None)
