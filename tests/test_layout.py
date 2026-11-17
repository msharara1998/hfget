import time
from pathlib import Path

import pytest

from hfget.layout import CacheLayout, repo_cache_folder

CONTENT = b'{"hidden_size": 1024}'


@pytest.fixture
def file(remote_file):
    return remote_file("sub/dir/config.json", CONTENT)


def write_partial(layout, file, content=CONTENT):
    partial = layout.partial_path(file)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(content)
    return partial


# LocalDirLayout


def test_local_dir_paths(local_layout, file):
    root = local_layout.root
    downloads = root / ".cache" / "huggingface" / "download"
    assert local_layout.final_path(file) == root / "sub" / "dir" / "config.json"
    assert local_layout.partial_path(file) == downloads / "sub" / "dir" / "config.json.incomplete"


def test_local_dir_prepare_writes_gitignore(local_layout):
    local_layout.prepare()
    gitignore = local_layout.root / ".cache" / "huggingface" / ".gitignore"
    assert gitignore.read_text() == "*\n"


def test_local_dir_finish_moves_file_and_writes_metadata(local_layout, file):
    partial = write_partial(local_layout, file)
    before = time.time()
    local_layout.finish(file)
    after = time.time()

    assert not partial.exists()
    assert local_layout.final_path(file).read_bytes() == CONTENT
    metadata = partial.with_name("config.json.metadata")
    commit, etag, stamp, end = metadata.read_text().split("\n")
    assert (commit, etag, end) == (local_layout.commit, file.etag, "")
    assert before <= float(stamp) <= after


def test_local_dir_is_done(local_layout, file, remote_file):
    assert not local_layout.is_done(file)
    write_partial(local_layout, file)
    local_layout.finish(file)
    assert local_layout.is_done(file)

    same_size_other_etag = remote_file(file.path, b"x" * len(CONTENT))
    assert not local_layout.is_done(same_size_other_etag)

    local_layout.final_path(file).write_bytes(CONTENT + b"\n")
    assert not local_layout.is_done(file)


def test_local_dir_is_not_done_without_metadata(local_layout, file):
    final = local_layout.final_path(file)
    final.parent.mkdir(parents=True)
    final.write_bytes(CONTENT)
    assert not local_layout.is_done(file)


# CacheLayout


def test_cache_paths(cache_layout, file):
    storage = cache_layout.root.parent.parent
    assert cache_layout.root == storage / "snapshots" / cache_layout.commit
    assert cache_layout.final_path(file) == storage / "blobs" / file.etag
    assert cache_layout.partial_path(file) == storage / "blobs" / f"{file.etag}.incomplete"


def test_cache_finish_moves_blob_and_links_it(cache_layout, file):
    partial = write_partial(cache_layout, file)
    cache_layout.finish(file)

    blob = cache_layout.final_path(file)
    pointer = cache_layout.root / file.path
    assert not partial.exists()
    assert blob.read_bytes() == CONTENT
    assert pointer.is_symlink()
    target = pointer.readlink()
    assert not target.is_absolute()
    assert target == Path("../../../../blobs") / file.etag
    assert pointer.resolve() == blob.resolve()
    assert pointer.read_bytes() == CONTENT


def test_cache_is_done_checks_the_blob_size(cache_layout, file):
    assert not cache_layout.is_done(file)
    write_partial(cache_layout, file)
    cache_layout.finish(file)
    assert cache_layout.is_done(file)
    cache_layout.final_path(file).write_bytes(b"short")
    assert not cache_layout.is_done(file)


def test_missing_empty_file_is_not_done(local_layout, cache_layout, remote_file):
    # A missing file has size 0, the same as an empty file: it must still count as missing.
    empty = remote_file("empty.txt")
    assert not cache_layout.is_done(empty)
    write_partial(local_layout, empty, b"")
    local_layout.finish(empty)
    assert local_layout.is_done(empty)
    local_layout.final_path(empty).unlink()  # the record stays, the file is gone
    assert not local_layout.is_done(empty)


def test_cache_finalize_links_files_and_writes_ref(cache_layout, file):
    write_partial(cache_layout, file)
    cache_layout.finish(file)
    (cache_layout.root / file.path).unlink()

    cache_layout.finalize([file])

    storage = cache_layout.root.parent.parent
    assert (storage / "refs" / "main").read_text() == cache_layout.commit
    assert (cache_layout.root / file.path).read_bytes() == CONTENT


def test_cache_finalize_writes_ref_of_named_revision(tmp_path):
    storage = tmp_path / "models--org--name"
    commit = "a" * 40
    CacheLayout(storage, commit, "v1.0").finalize([])
    assert (storage / "refs" / "v1.0").read_text() == commit
    assert not (storage / "refs" / "main").exists()


def test_cache_finalize_writes_no_ref_for_a_commit(tmp_path):
    storage = tmp_path / "models--org--name"
    commit = "a" * 40
    CacheLayout(storage, commit, commit).finalize([])
    assert not (storage / "refs").exists()


@pytest.mark.parametrize(
    ("repo_type", "folder"),
    [("model", "models--Qwen--Qwen3-0.6B"), ("dataset", "datasets--Qwen--Qwen3-0.6B")],
)
def test_repo_cache_folder(tmp_path, repo_type, folder):
    assert repo_cache_folder("Qwen/Qwen3-0.6B", repo_type, tmp_path) == tmp_path / folder
