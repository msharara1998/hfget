"""The download loop, driven through `execute_plan` with a fake curl on PATH.

The fake (`fake_curl.py`) writes bytes of a known payload into the partial file and exits
with the codes each test asks for, so retries, resumes and checksums run for real.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from hfget import downloader
from hfget.api import DownloadPlan, execute_plan
from hfget.downloader import _Backoff
from hfget.errors import DownloadError
from hfget.progress import DownloadProgress, FileState

TOKEN = "hf_test_token"
PAYLOAD = bytes(range(256)) * 4  # 1 KiB
RESET = {"append": 0, "exit": 56, "stderr": "curl: (56) Recv failure: Connection reset by peer"}


class FakeCurl:
    """Serves files to the fake curl script and reads back what it saw."""

    def __init__(self, state, make_file):
        self._state = state
        self._make_file = make_file
        self._count = 0

    def serve(self, path, content=PAYLOAD, steps=(), *, lfs=True):
        """Return a repo file whose URL makes the fake curl follow `steps`."""
        file_id = f"file{self._count}"
        self._count += 1
        (self._state / f"{file_id}.bin").write_bytes(content)
        (self._state / f"{file_id}.json").write_text(json.dumps(list(steps)))
        return self._make_file(path, content, lfs=lfs, url=f"https://hub.test/{file_id}")

    def calls(self, file):
        """One dict per curl run for this file: argv, stdin, size_before."""
        log = self._state / f"{file.url.rsplit('/', 1)[-1]}.calls"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    def sizes_before(self, file):
        return [call["size_before"] for call in self.calls(file)]


@pytest.fixture
def fake_curl(tmp_path, monkeypatch, remote_file):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "curl"
    source = Path(__file__).with_name("fake_curl.py").read_text()
    script.write_text(f"#!{sys.executable}\n{source}")
    script.chmod(0o755)
    state = tmp_path / "curl-state"
    state.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_CURL_STATE", str(state))
    monkeypatch.setattr(downloader, "MAX_WAIT_SECONDS", 0)  # retry at once
    return FakeCurl(state, remote_file)


@pytest.fixture(params=["local", "cache"])
def layout(request):
    return request.getfixturevalue(f"{request.param}_layout")


def run(layout, files, on_progress=None):
    """Plan the files by hand, as `plan_download` would, and run the plan."""
    layout.prepare()
    pending = tuple(file for file in files if not layout.is_done(file))
    plan = DownloadPlan(
        repo_id="org/name",
        repo_type="model",
        commit=layout.commit,
        files=tuple(files),
        pending=pending,
        bytes_to_download=sum(file.size for file in pending),
        path=layout.root,
        layout=layout,
    )
    return execute_plan(plan, token=TOKEN, max_workers=4, on_progress=on_progress)


def write_partial(layout, file, content):
    partial = layout.partial_path(file)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(content)


def downloaded(layout, file):
    """The finished file as a user sees it: `<root>/<path>` (a symlink in the cache)."""
    return (layout.root / file.path).read_bytes()


@pytest.mark.parametrize("lfs", [True, False], ids=["lfs", "git"])
def test_fresh_download(fake_curl, layout, lfs):
    files = [
        fake_curl.serve("config.json", b'{"a": 1}', lfs=lfs),
        fake_curl.serve("weights/model.bin", PAYLOAD, lfs=lfs),
    ]

    assert run(layout, files) == layout.root

    for file in files:
        assert not layout.partial_path(file).exists()
        assert layout.is_done(file)
        [call] = fake_curl.calls(file)
        assert call["size_before"] == 0
        assert call["argv"][-1] == file.url
        assert all(TOKEN not in arg for arg in call["argv"])
        assert call["stdin"] == f'header = "Authorization: Bearer {TOKEN}"\n'
    assert downloaded(layout, files[0]) == b'{"a": 1}'
    assert downloaded(layout, files[1]) == PAYLOAD


def test_cache_layout_writes_ref(fake_curl, cache_layout):
    run(cache_layout, [fake_curl.serve("model.bin")])
    storage = cache_layout.root.parent.parent
    assert (storage / "refs" / "main").read_text() == cache_layout.commit


def test_identical_blobs_in_cache_are_fetched_once(fake_curl, cache_layout):
    first = fake_curl.serve("a/model.bin")
    second = fake_curl.serve("b/model.bin")  # same content, so same etag and blob
    run(cache_layout, [first, second])
    assert len(fake_curl.calls(first)) == 1
    assert fake_curl.calls(second) == []
    assert downloaded(cache_layout, first) == downloaded(cache_layout, second) == PAYLOAD


def test_resume_continues_partial_file(fake_curl, layout):
    file = fake_curl.serve("model.bin")
    write_partial(layout, file, PAYLOAD[:300])

    run(layout, [file])

    [call] = fake_curl.calls(file)
    assert call["size_before"] == 300
    assert call["argv"][call["argv"].index("--continue-at") + 1] == "-"
    assert downloaded(layout, file) == PAYLOAD


def test_partial_bigger_than_file_starts_over(fake_curl, local_layout):
    file = fake_curl.serve("model.bin")
    write_partial(local_layout, file, PAYLOAD + b"extra")
    run(local_layout, [file])
    assert fake_curl.sizes_before(file) == [0]
    assert downloaded(local_layout, file) == PAYLOAD


def test_retry_after_progress_resets_failure_count(fake_curl, local_layout, monkeypatch):
    # With MAX_FAILURES = 1, two failures with no progress in a row would give up.
    # The progress in the second run resets the count, so the download completes.
    monkeypatch.setattr(downloader, "MAX_FAILURES", 1)
    progress = {**RESET, "append": 100}
    file = fake_curl.serve("model.bin", steps=[RESET, progress, RESET, {}])
    snapshots = []

    run(local_layout, [file], on_progress=snapshots.append)

    assert fake_curl.sizes_before(file) == [0, 0, 100, 100]
    assert downloaded(local_layout, file) == PAYLOAD
    assert snapshots[-1].retries == 3


def test_gives_up_after_too_many_failures_without_progress(fake_curl, local_layout, monkeypatch):
    monkeypatch.setattr(downloader, "MAX_FAILURES", 2)
    file = fake_curl.serve("model.bin", steps=[RESET])

    with pytest.raises(DownloadError, match=r"gave up after 3 attempts.*Connection reset"):
        run(local_layout, [file])

    assert len(fake_curl.calls(file)) == 3
    assert local_layout.partial_path(file).exists()  # kept for the next run
    assert not local_layout.final_path(file).exists()


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"exit": 22, "http": "404", "stderr": "curl: (22) error: 404"}, "HTTP 404"),
        ({"exit": 22, "http": "401"}, "HTTP 401"),
        ({"append": 10, "exit": 23, "stderr": "curl: (23) Failure writing"}, "could not write"),
    ],
    ids=["http-404", "http-401", "write-error"],
)
def test_fatal_errors_raise_at_once(fake_curl, local_layout, step, message):
    file = fake_curl.serve("model.bin", steps=[step])
    with pytest.raises(DownloadError, match=message):
        run(local_layout, [file])
    assert len(fake_curl.calls(file)) == 1


def test_server_error_is_retried(fake_curl, local_layout):
    file = fake_curl.serve("model.bin", steps=[{"exit": 22, "http": "500"}, {}])
    run(local_layout, [file])
    assert len(fake_curl.calls(file)) == 2
    assert downloaded(local_layout, file) == PAYLOAD


def test_range_error_deletes_partial_and_starts_over(fake_curl, local_layout):
    range_error = {"append": 0, "exit": 33, "stderr": "curl: (33) server does not support ranges"}
    file = fake_curl.serve("model.bin", steps=[range_error, {}])
    write_partial(local_layout, file, b"stale bytes from another version")

    run(local_layout, [file])

    assert fake_curl.sizes_before(file) == [32, 0]
    assert downloaded(local_layout, file) == PAYLOAD


@pytest.mark.parametrize("lfs", [True, False], ids=["lfs", "git"])
def test_checksum_mismatch_downloads_again(fake_curl, local_layout, lfs):
    file = fake_curl.serve("model.bin", steps=[{"corrupt": True}, {}], lfs=lfs)
    run(local_layout, [file])
    assert fake_curl.sizes_before(file) == [0, 0]
    assert downloaded(local_layout, file) == PAYLOAD


def test_checksum_keeps_failing(fake_curl, local_layout):
    file = fake_curl.serve("model.bin", steps=[{"corrupt": True}])

    with pytest.raises(DownloadError, match="checksum keeps failing"):
        run(local_layout, [file])

    assert len(fake_curl.calls(file)) == downloader.MAX_CHECKSUM_MISMATCHES + 1
    assert not local_layout.partial_path(file).exists()
    assert not local_layout.final_path(file).exists()


def test_empty_file_needs_no_curl_run(fake_curl, layout):
    file = fake_curl.serve("empty.txt", b"", lfs=False)
    run(layout, [file])
    assert fake_curl.calls(file) == []
    assert downloaded(layout, file) == b""
    assert layout.is_done(file)


def test_progress_snapshots(fake_curl, local_layout):
    resumed = fake_curl.serve("model.bin")
    fresh = fake_curl.serve("config.json", b'{"a": 1}', lfs=False)
    write_partial(local_layout, resumed, PAYLOAD[:300])
    snapshots = []

    run(local_layout, [resumed, fresh], on_progress=snapshots.append)

    assert all(isinstance(snapshot, DownloadProgress) for snapshot in snapshots)
    first, last = snapshots[0], snapshots[-1]
    assert [file.path for file in first.files] == ["model.bin", "config.json"]
    assert [file.state for file in first.files] == [FileState.QUEUED, FileState.QUEUED]
    assert [file.bytes_on_disk for file in first.files] == [300, 0]
    assert [file.state for file in last.files] == [FileState.DONE, FileState.DONE]
    assert first.started_bytes == last.started_bytes == 300
    assert first.total_bytes == last.done_bytes == last.total_bytes == len(PAYLOAD) + 8
    assert last.retries == 0
    timestamps = [snapshot.timestamp for snapshot in snapshots]
    assert timestamps == sorted(timestamps)


def test_backoff_policy(monkeypatch):
    monkeypatch.setattr(downloader, "MAX_FAILURES", 3)
    backoff = _Backoff()
    waits = []
    for _ in range(3):
        backoff.record(progressed=False)
        waits.append(backoff.wait_seconds)
    assert waits == [2, 4, 8]
    assert not backoff.exhausted
    backoff.record(progressed=True)
    assert (backoff.failures, backoff.wait_seconds) == (0, 1)
    for _ in range(4):
        backoff.record(progressed=False)
    assert backoff.exhausted
    backoff.failures = 20
    assert backoff.wait_seconds == downloader.MAX_WAIT_SECONDS
