"""Real downloads from the Hugging Face Hub. Run with `pytest -m network`.

Only one small file (about 700 bytes) is downloaded, because the network can be slow.
"""

import json

import pytest

from hfget import download

pytestmark = pytest.mark.network

REPO = "Qwen/Qwen3-0.6B"


def test_download_into_local_dir(tmp_path):
    path = download(REPO, filenames=["config.json"], local_dir=tmp_path)

    assert path == tmp_path / "config.json"
    assert isinstance(json.loads(path.read_text()), dict)
    bookkeeping = tmp_path / ".cache" / "huggingface"
    assert (bookkeeping / ".gitignore").read_text() == "*\n"
    commit, etag, _stamp = (bookkeeping / "download" / "config.json.metadata").read_text().split()
    assert len(commit) == 40
    assert len(etag) == 40  # config.json is not an LFS file: its etag is a git blob id
    assert not (bookkeeping / "download" / "config.json.incomplete").exists()


def test_download_into_cache(tmp_path):
    path = download(REPO, filenames=["config.json"], cache_dir=tmp_path)

    storage = tmp_path / "models--Qwen--Qwen3-0.6B"
    commit = (storage / "refs" / "main").read_text()
    assert path == storage / "snapshots" / commit / "config.json"
    assert path.is_symlink()
    assert isinstance(json.loads(path.read_text()), dict)
    [blob] = (storage / "blobs").iterdir()
    assert path.resolve() == blob.resolve()
