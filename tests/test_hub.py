import logging

import pytest

from hfget.errors import DownloadError
from hfget.hub import select_files, split_repo_id

REPO_PATHS = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "onnx/model.onnx",
    "onnx/model_q4.onnx",
    "README.md",
)


@pytest.fixture
def repo(remote_file):
    return [remote_file(path) for path in REPO_PATHS]


def paths(files):
    return [file.path for file in files]


@pytest.mark.parametrize(
    ("raw", "default_type", "expected"),
    [
        ("org/name", "model", ("org/name", "model")),
        ("org/name", "dataset", ("org/name", "dataset")),
        ("gpt2", "model", ("gpt2", "model")),
        ("datasets/org/name", "model", ("org/name", "dataset")),
        ("spaces/org/name", "model", ("org/name", "space")),
        ("models/org/name", "dataset", ("org/name", "model")),
        ("kernels/org/name", "model", ("org/name", "kernel")),
        ("hf://org/name", "model", ("org/name", "model")),
        ("hf://datasets/org/name", "model", ("org/name", "dataset")),
    ],
)
def test_split_repo_id(raw, default_type, expected):
    assert split_repo_id(raw, default_type) == expected


def test_select_all_files_without_filters(repo):
    assert paths(select_files(repo, [], None, None)) == list(REPO_PATHS)


@pytest.mark.parametrize(
    ("include", "exclude", "expected"),
    [
        (["*.json"], None, ["config.json", "tokenizer.json"]),
        (None, ["onnx/*"], ["config.json", "model.safetensors", "tokenizer.json", "README.md"]),
        (["*.json", "*.safetensors"], ["tokenizer*"], ["config.json", "model.safetensors"]),
        (["onnx/"], None, ["onnx/model.onnx", "onnx/model_q4.onnx"]),
    ],
)
def test_select_with_patterns(repo, include, exclude, expected):
    assert paths(select_files(repo, [], include, exclude)) == expected


def test_filenames_override_patterns_with_a_warning(repo, caplog):
    with caplog.at_level(logging.WARNING, logger="hfget"):
        selected = select_files(repo, ["config.json"], ["*.onnx"], ["*.json"])
    assert paths(selected) == ["config.json"]
    assert "Ignoring --include and --exclude" in caplog.text


def test_filenames_without_patterns_log_nothing(repo, caplog):
    with caplog.at_level(logging.WARNING, logger="hfget"):
        select_files(repo, ["config.json"], None, None)
    assert caplog.records == []


def test_filenames_keep_repo_order(repo):
    assert paths(select_files(repo, ["tokenizer.json", "config.json"], None, None)) == [
        "config.json",
        "tokenizer.json",
    ]


def test_filename_ending_with_slash_selects_a_folder(repo):
    selected = select_files(repo, ["onnx/"], None, None)
    assert paths(selected) == ["onnx/model.onnx", "onnx/model_q4.onnx"]


def test_folder_or_pattern_that_matches_nothing_is_not_an_error(repo):
    assert select_files(repo, ["missing/", "*.bin"], None, None) == []


def test_missing_exact_filename_raises(repo):
    with pytest.raises(DownloadError, match=r"Not found in the repo: missing\.bin, other\.txt"):
        select_files(repo, ["config.json", "missing.bin", "other.txt"], None, None)


@pytest.mark.parametrize("lfs", [True, False], ids=["lfs-sha256", "git-blob-sha1"])
def test_checksum_matches(tmp_path, remote_file, lfs):
    content = b"some weights"
    file = remote_file("weights.bin", content, lfs=lfs)
    local = tmp_path / "weights.bin"
    local.write_bytes(content)
    assert file.checksum_matches(local)
    local.write_bytes(b"some weightz")
    assert not file.checksum_matches(local)


def test_git_blob_id_of_empty_file(remote_file):
    # The well-known git id of an empty blob, to check the test helper itself.
    assert remote_file("empty.txt").etag == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
