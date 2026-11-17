"""The live display never wraps: every line fits in `width - 1` cells and `height - 1` rows."""

import os

import pytest

from hfget.api import DownloadPlan
from hfget.frames import build_done_line, build_frame, build_header, build_summary
from hfget.layout import LocalDirLayout
from hfget.progress import DownloadProgress, FileProgress, FileState, RetryInfo
from hfget.speed import Stats
from hfget.style import (
    ASCII_GLYPHS,
    UNICODE_GLYPHS,
    Theme,
    format_byte_pair,
    format_bytes,
    format_duration,
    format_speed,
    visible_width,
)
from hfget.utils import human_size

GB = 1_000_000_000
NOW = 1000.0
LONG_NAME = "model-with-a-really-long-descriptive-name/" * 3 + "model-00001-of-00042.safetensors"
WIDE_NAME = "モデル/重み-ファイル-0001.safetensors"  # double-width characters

THEMES = [Theme(UNICODE_GLYPHS, color=True), Theme(ASCII_GLYPHS, color=False)]
WIDTHS = range(20, 201, 9)
HEIGHTS = [2, 3, 5, 12, 50]


def file_progress(path, state, size=5 * GB, done=2 * GB, reason="connection reset"):
    retry = RetryInfo(12, NOW + 3725, reason) if state is FileState.RETRYING else None
    return FileProgress(path=path, size=size, bytes_on_disk=done, state=state, retry=retry)


def snapshot(files):
    return DownloadProgress(
        files=tuple(files),
        started_bytes=GB,
        done_bytes=sum(file.bytes_on_disk for file in files),
        total_bytes=sum(file.size for file in files),
        retries=7,
        timestamp=NOW,
    )


SNAPSHOTS = {
    "every-state": snapshot(
        [
            file_progress("config.json", FileState.DONE, size=726, done=726),
            file_progress("model.safetensors", FileState.DOWNLOADING),
            file_progress(LONG_NAME, FileState.RETRYING, reason="no data for 60 s" * 4),
            file_progress(WIDE_NAME, FileState.VERIFYING, done=5 * GB),
            file_progress("tokenizer.json", FileState.QUEUED, done=0),
            file_progress("huge.bin", FileState.DOWNLOADING, size=999 * 1000 * GB, done=GB),
        ]
    ),
    "many-files": snapshot(
        [file_progress(f"shard-{i:05d}-of-00060.bin", FileState.DOWNLOADING) for i in range(30)]
        + [file_progress(f"{LONG_NAME}.{i}", FileState.RETRYING) for i in range(10)]
        + [file_progress(f"queued-{i}", FileState.QUEUED, done=0) for i in range(20)]
    ),
    "all-queued": snapshot([file_progress(LONG_NAME, FileState.QUEUED, done=0)] * 3),
    "all-done": snapshot([file_progress(WIDE_NAME, FileState.DONE, done=5 * GB)]),
    "empty-names": snapshot([file_progress("", FileState.DOWNLOADING, size=0, done=0)]),
}
STATS = [
    Stats(speed=None, file_speeds={}, elapsed=0.0),
    Stats(
        speed=123_456_789.0,
        file_speeds={"model.safetensors": 98_765_432.0, "huge.bin": 0.5},
        elapsed=98_765.0,
    ),
    Stats(speed=0.01, file_speeds={}, elapsed=5.0),  # an ETA of years
]


@pytest.fixture
def plan(tmp_path, remote_file):
    files = (remote_file("config.json"), remote_file(LONG_NAME))
    layout = LocalDirLayout(tmp_path / LONG_NAME, "0123abcd" * 5)
    return DownloadPlan(
        repo_id="some-organization/" + "a-very-long-repository-name" * 2,
        repo_type="model",
        commit=layout.commit,
        files=files,
        pending=files[1:],
        bytes_to_download=0,
        path=layout.root,
        layout=layout,
    )


def assert_fits(lines, width):
    for line in lines:
        assert visible_width(line) <= width - 1, (width, line)


@pytest.mark.parametrize("theme", THEMES, ids=["unicode-color", "ascii-plain"])
@pytest.mark.parametrize("name", SNAPSHOTS)
def test_frame_fits_the_terminal(name, theme):
    progress = SNAPSHOTS[name]
    for stats in STATS:
        for width in WIDTHS:
            for height in HEIGHTS:
                frame = build_frame(progress, stats, os.terminal_size((width, height)), theme)
                assert_fits(frame, width)
                assert 1 <= len(frame) <= height - 1, (width, height, frame)


@pytest.mark.parametrize("theme", THEMES, ids=["unicode-color", "ascii-plain"])
def test_permanent_lines_fit_the_terminal(plan, theme):
    progress = SNAPSHOTS["every-state"]
    for width in WIDTHS:
        assert_fits(build_header(plan, progress, width, theme), width)
        assert_fits([build_summary(progress, STATS[1], width, theme)], width)
        for file in progress.files:
            assert_fits([build_done_line(file, width, theme)], width)


def test_frame_shows_hidden_and_queued_files():
    # 6 rows leave 5 lines: 3 of the 40 active files, the "more" row and the total row.
    frame = build_frame(SNAPSHOTS["many-files"], STATS[0], os.terminal_size((120, 6)), THEMES[1])
    assert len(frame) == 5
    assert frame[-2].strip() == "+ 37 more active - 20 queued"
    assert frame[-1].strip().startswith("Total 0/60")


@pytest.mark.parametrize(
    ("size", "text"),
    [(0, "0.0"), (999, "999.0"), (1570, "1.6K"), (2_500_000, "2.5M"), (5.3 * GB, "5.3G")],
)
def test_human_size(size, text):
    assert human_size(size) == text


@pytest.mark.parametrize(
    ("size", "text"),
    [(512, "512 B"), (3_200, "3.20 kB"), (999_999, "1.00 MB"), (3.98 * GB, "3.98 GB")],
)
def test_format_bytes(size, text):
    assert format_bytes(size) == text


def test_format_byte_pair():
    assert format_byte_pair(3.91 * GB, 9.32 * GB) == "3.91 / 9.32 GB"
    assert format_byte_pair(640_000_000, 1.5 * GB) == "640 MB / 1.50 GB"


def test_format_speed():
    assert format_speed(None) == "--"
    assert format_speed(2_100_000) == "2.1 MB/s"
    assert format_speed(-5) == "0 B/s"


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "0 s"),
        (-3, "0 s"),
        (0.3, "1 s"),
        (42, "42 s"),
        (185, "3 min 05 s"),
        (600, "10 min"),
        (3900, "1 h 05 min"),
    ],
)
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text
