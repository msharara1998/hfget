"""Pure functions that build the lines of the progress display.

Used only by the command line display. Nothing here writes to the terminal, so every
function can be tested with fake snapshots. Every line a function returns is at most
`width - 1` cells wide, so the terminal never wraps it.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .api import DownloadPlan
from .progress import DownloadProgress, FileProgress, FileState
from .speed import Stats
from .style import (
    Glyphs,
    Segment,
    Style,
    Theme,
    cell_width,
    format_byte_pair,
    format_bytes,
    format_duration,
    format_speed,
    pad,
    plural,
    take_prefix,
    truncate_middle,
)

ACTIVE_STATES = (FileState.DOWNLOADING, FileState.RETRYING, FileState.VERIFYING)
SEP = "  "  # between columns
ICON_WIDTH = 2  # the icon and one space
NAME_PREFERRED = 16  # name width kept before optional columns are dropped
NAME_MIN = 8  # a status row may shrink the name down to this width
STATUS_MIN = 18  # the room a status row tries to keep for its text
NAME_MAX = 60
BAR_MIN = 10
BAR_MAX = 40
SPINNER_FRAMES_PER_SECOND = 10


class Column(Enum):
    """A column after the file name. `PERCENT` is never dropped."""

    BAR = "bar"
    PERCENT = "percent"
    BYTES = "bytes"
    SPEED = "speed"
    ETA = "eta"  # total row only
    ELAPSED = "elapsed"  # total row only


COLUMN_ORDER = tuple(Column)
COLUMN_WIDTHS = {
    Column.BAR: BAR_MIN,  # the bar can grow past this minimum
    Column.PERCENT: 4,  # "100%"
    Column.BYTES: 17,  # "1.50 GB / 1.50 TB"
    Column.SPEED: 9,  # "12.3 MB/s"
    Column.ETA: 15,  # "ETA 3 min 05 s"
    Column.ELAPSED: 18,  # "elapsed 3 min 05 s"
}
DROP_ORDER = (Column.ELAPSED, Column.SPEED, Column.ETA, Column.BYTES, Column.BAR)


@dataclass(frozen=True)
class Layout:
    """The column widths of one frame, shared by all its rows.

    Attributes:
        name: The width of the name column.
        bar: The width of the bar, or 0 if the bar is hidden.
        columns: The columns shown after the name.
    """

    name: int
    bar: int
    columns: frozenset[Column]


def compute_layout(width: int, natural_name: int) -> Layout:
    """Fit the columns into a width, dropping optional columns as it shrinks.

    Columns are dropped in `DROP_ORDER` until the name keeps a readable width. Below that,
    the name shrinks. Spare room goes first to the name, then to the bar.

    Args:
        width: The usable width in cells (the terminal width minus 1).
        natural_name: The width of the longest file name.

    Returns:
        The layout.
    """
    columns = set(COLUMN_ORDER)
    preferred = min(natural_name, NAME_PREFERRED)

    def needed() -> int:
        """Return the width of a row with the current columns and the preferred name."""
        return ICON_WIDTH + preferred + sum(len(SEP) + COLUMN_WIDTHS[c] for c in columns)

    for column in DROP_ORDER:
        if needed() <= width:
            break
        columns.discard(column)

    space = width - ICON_WIDTH - sum(len(SEP) + COLUMN_WIDTHS[c] for c in columns)
    bar = 0
    if Column.BAR in columns:
        space += BAR_MIN  # the bar shares this space with the name
        name = max(1, min(natural_name, NAME_MAX, space - BAR_MIN))
        bar = min(BAR_MAX, space - name)
    else:
        name = max(1, min(natural_name, NAME_MAX, space))
    return Layout(name=name, bar=bar, columns=frozenset(columns))


def build_header(
    plan: DownloadPlan, snapshot: DownloadProgress, width: int, theme: Theme
) -> list[str]:
    """Build the two lines printed once when the download starts.

    Example:
        Qwen/Qwen3.5-4B @ 851bf6e8 → models/Qwen3.5-4B
        2 files to download (2 resuming, 3.98 GB already on disk) · 12 already complete

    Args:
        plan: The download plan.
        snapshot: The first progress snapshot.
        width: The terminal width.
        theme: The glyphs and colors.

    Returns:
        The two lines.
    """
    usable = max(1, width - 1)
    glyphs = theme.glyphs
    prefix: list[Segment] = [
        (plan.repo_id, (Style.BOLD,)),
        (f" @ {plan.commit[:8]} {glyphs.arrow} ", (Style.DIM,)),
    ]
    room = usable - sum(cell_width(text) for text, _ in prefix)
    destination = str(plan.layout.root)
    if room >= NAME_PREFERRED:
        destination = truncate_middle(destination, room, glyphs.ellipsis)

    resuming = sum(1 for file in snapshot.files if file.bytes_on_disk > 0)
    summary = f"{plural(len(snapshot.files), 'file')} to download"
    if resuming:
        on_disk = format_bytes(snapshot.started_bytes)
        summary += f" ({resuming} resuming, {on_disk} already on disk)"
    complete = len(plan.files) - len(plan.pending)
    if complete:
        summary += f" {glyphs.dot} {complete} already complete"
    return [
        theme.render([*prefix, (destination, ())], usable),
        theme.render([(summary, (Style.DIM,))], usable),
    ]


def build_frame(
    snapshot: DownloadProgress, stats: Stats, size: os.terminal_size, theme: Theme
) -> list[str]:
    """Build the live area: one row per active file, a queue row, and the total row.

    Args:
        snapshot: The newest snapshot.
        stats: The speeds and elapsed time for this snapshot.
        size: The terminal size. The frame has at most `size.lines - 1` lines.
        theme: The glyphs and colors.

    Returns:
        The lines of the live area. The total row is always the last one.
    """
    usable = max(1, size.columns - 1)
    natural_name = max((cell_width(file.path) for file in snapshot.files), default=0)
    rows = _RowBuilder(compute_layout(usable, natural_name), theme, stats, snapshot.timestamp)
    active = [file for file in snapshot.files if file.state in ACTIVE_STATES]
    queued = sum(1 for file in snapshot.files if file.state is FileState.QUEUED)
    room = max(1, size.lines - 1) - 1  # lines left once the total row is placed
    shown, hidden = _split_rows(active, queued, room)
    lines = [rows.file_row(file, usable) for file in shown]
    if (hidden or queued) and room > len(shown):
        lines.append(rows.more_row(hidden, queued, usable))
    lines.append(rows.total_row(snapshot, usable))
    return lines


def build_done_line(file: FileProgress, width: int, theme: Theme) -> str:
    """Build the permanent line of a finished file, for example "✓ config.json  3.2 kB".

    Args:
        file: The finished file.
        width: The terminal width.
        theme: The glyphs and colors.

    Returns:
        The line.
    """
    usable = max(1, width - 1)
    size = format_bytes(file.size)
    room = usable - ICON_WIDTH - len(SEP) - cell_width(size)
    name = truncate_middle(file.path, max(1, room), theme.glyphs.ellipsis)
    segments: list[Segment] = [
        (theme.glyphs.done, (Style.GREEN,)),
        (" ", ()),
        (name, ()),
        (SEP, ()),
        (size, (Style.DIM,)),
    ]
    return theme.render(segments, usable)


def build_summary(snapshot: DownloadProgress, stats: Stats, width: int, theme: Theme) -> str:
    """Build the line printed when every file is done.

    Example:
        ✓ Downloaded 2 files (5.34 GB) in 42 min · average 2.1 MB/s · 1 retry

    Args:
        snapshot: The last snapshot.
        stats: The stats of the last snapshot.
        width: The terminal width.
        theme: The glyphs and colors.

    Returns:
        The line.
    """
    glyphs = theme.glyphs
    files = plural(len(snapshot.files), "file")
    size = format_bytes(snapshot.total_bytes)
    segments: list[Segment] = [
        (f"{glyphs.done} ", (Style.GREEN, Style.BOLD)),
        (f"Downloaded {files} ({size}) in {format_duration(stats.elapsed)}", ()),
        (f" {glyphs.dot} average {format_speed(stats.average_speed(snapshot))}", (Style.DIM,)),
    ]
    if snapshot.retries:
        retries = plural(snapshot.retries, "retry", "retries")
        segments.append((f" {glyphs.dot} {retries}", (Style.YELLOW,)))
    return theme.render(segments, max(1, width - 1))


def build_plain_line(snapshot: DownloadProgress, stats: Stats, glyphs: Glyphs) -> str:
    """Build a progress line without colors, for logs and other non-terminal output.

    Example:
        [ 42%] 3.91 / 9.32 GB · 2.1 MB/s · ETA 43 min

    Args:
        snapshot: The newest snapshot.
        stats: The stats of this snapshot.
        glyphs: The glyphs (only the separator is used).

    Returns:
        The line.
    """
    percent = percent_of(snapshot.done_bytes, snapshot.total_bytes)
    sizes = format_byte_pair(snapshot.done_bytes, snapshot.total_bytes)
    return (
        f"[{percent:3d}%] {sizes} {glyphs.dot} {format_speed(stats.speed)} "
        f"{glyphs.dot} ETA {_format_eta(stats.eta(snapshot))}"
    )


def percent_of(done: int, total: int) -> int:
    """Return how far a count is, as a whole percent rounded down.

    Args:
        done: The part that is done.
        total: The whole. 0 counts as complete.

    Returns:
        A number from 0 to 100.
    """
    if total <= 0:
        return 100
    return max(0, min(100, done * 100 // total))


def _format_eta(seconds: float | None) -> str:
    """Format an estimated time left.

    Args:
        seconds: The time left, or None if unknown.

    Returns:
        For example "43 min", or "--" if unknown.
    """
    return "--" if seconds is None else format_duration(seconds)


def _split_rows(
    active: Sequence[FileProgress], queued: int, room: int
) -> tuple[list[FileProgress], int]:
    """Choose the file rows that fit, keeping one line for the queue row if needed.

    Args:
        active: The files that download, retry or are checked, in repo order.
        queued: How many files wait in the queue.
        room: How many lines are free for file rows and the queue row.

    Returns:
        The files to show, and how many active files are hidden.
    """
    if len(active) + (1 if queued else 0) <= room:
        return list(active), 0
    keep = max(0, room - 1)
    return list(active[:keep]), len(active) - keep


@dataclass(frozen=True)
class _RowBuilder:
    """Builds the rows of one frame with a shared layout.

    Attributes:
        layout: The column widths.
        theme: The glyphs and colors.
        stats: The speeds and elapsed time.
        now: The snapshot time, which drives the spinner and the retry countdown.
    """

    layout: Layout
    theme: Theme
    stats: Stats
    now: float

    def file_row(self, file: FileProgress, width: int) -> str:
        """Build the row of an active file.

        Args:
            file: A file that downloads, retries or is being checked.
            width: The usable width.

        Returns:
            The row.
        """
        icon, icon_style = self._icon(file.state)
        if file.state is not FileState.DOWNLOADING:
            text, style = self._status(file)
            # A narrow screen gives the status text room by shrinking the name.
            name_width = min(
                self.layout.name, max(NAME_MIN, width - ICON_WIDTH - len(SEP) - STATUS_MIN)
            )
            name = truncate_middle(file.path, name_width, self.theme.glyphs.ellipsis)
            segments: list[Segment] = [
                (icon, (icon_style,)),
                (" ", ()),
                (pad(name, name_width), ()),
            ]
            return self.theme.render([*segments, (SEP, ()), (text, (style,))], width)

        name = truncate_middle(file.path, self.layout.name, self.theme.glyphs.ellipsis)
        speed = format_speed(self.stats.file_speeds.get(file.path))
        segments = [
            (icon, (icon_style,)),
            (" ", ()),
            (pad(name, self.layout.name), ()),
            *self._columns(
                {
                    Column.BAR: self._bar(file.bytes_on_disk, file.size, Style.CYAN),
                    Column.PERCENT: self._cell(Column.PERCENT, self._percent(file)),
                    Column.BYTES: self._cell(
                        Column.BYTES, format_byte_pair(file.bytes_on_disk, file.size), Style.DIM
                    ),
                    Column.SPEED: self._cell(Column.SPEED, speed, Style.DIM),
                }
            ),
        ]
        return self.theme.render(segments, width)

    def more_row(self, hidden: int, queued: int, width: int) -> str:
        """Build the row that counts the files not shown, for example "+ 12 queued".

        Args:
            hidden: Active files that did not fit.
            queued: Files waiting in the queue.
            width: The usable width.

        Returns:
            The row.
        """
        parts = []
        if hidden:
            parts.append(f"{hidden} more active")
        if queued:
            parts.append(f"{queued} queued")
        text = f"+ {f' {self.theme.glyphs.dot} '.join(parts)}"
        return self.theme.render([(" " * ICON_WIDTH, ()), (text, (Style.DIM,))], width)

    def total_row(self, snapshot: DownloadProgress, width: int) -> str:
        """Build the total row: bar, percent, bytes, speed, ETA and elapsed time.

        Args:
            snapshot: The newest snapshot.
            width: The usable width.

        Returns:
            The row.
        """
        done = sum(1 for file in snapshot.files if file.state is FileState.DONE)
        label = f"Total {done}/{len(snapshot.files)}"
        if cell_width(label) > self.layout.name:
            label = f"{done}/{len(snapshot.files)}"
        complete = snapshot.done_bytes >= snapshot.total_bytes
        sizes = format_byte_pair(snapshot.done_bytes, snapshot.total_bytes)
        percent = f"{percent_of(snapshot.done_bytes, snapshot.total_bytes)}%"
        eta = f"ETA {_format_eta(self.stats.eta(snapshot))}"
        elapsed = f"elapsed {format_duration(self.stats.elapsed)}"
        bar_style = Style.GREEN if complete else Style.CYAN
        segments: list[Segment] = [
            (" " * ICON_WIDTH, ()),
            (pad(take_prefix(label, self.layout.name), self.layout.name), (Style.BOLD,)),
            *self._columns(
                {
                    Column.BAR: self._bar(snapshot.done_bytes, snapshot.total_bytes, bar_style),
                    Column.PERCENT: self._cell(Column.PERCENT, percent, Style.BOLD),
                    Column.BYTES: self._cell(Column.BYTES, sizes, Style.DIM),
                    Column.SPEED: self._cell(Column.SPEED, format_speed(self.stats.speed)),
                    # Only the total row has these two, so they need no padding.
                    Column.ETA: [(take_prefix(eta, COLUMN_WIDTHS[Column.ETA]), ())],
                    Column.ELAPSED: [
                        (take_prefix(elapsed, COLUMN_WIDTHS[Column.ELAPSED]), (Style.DIM,))
                    ],
                }
            ),
        ]
        return self.theme.render(segments, width)

    def _columns(self, cells: dict[Column, list[Segment]]) -> list[Segment]:
        """Lay out the cells of a row in column order, keeping only the shown columns.

        Args:
            cells: The content of each column this row has.

        Returns:
            The segments, with separators between the columns.
        """
        segments: list[Segment] = []
        for column in COLUMN_ORDER:
            if column in self.layout.columns and column in cells:
                segments += [(SEP, ()), *cells[column]]
        return segments

    def _cell(self, column: Column, text: str, *styles: Style) -> list[Segment]:
        """Fit a text into a fixed-width column, aligned to the right.

        Args:
            column: The column.
            text: The text.
            *styles: The styles of the text.

        Returns:
            One segment exactly as wide as the column.
        """
        width = COLUMN_WIDTHS[column]
        return [(pad(take_prefix(text, width), width, right=True), styles)]

    def _bar(self, done: int, total: int, style: Style) -> list[Segment]:
        """Draw a progress bar with half-cell steps.

        Args:
            done: The part that is done.
            total: The whole. 0 counts as complete.
            style: The style of the filled part.

        Returns:
            The bar, exactly `layout.bar` cells wide.
        """
        glyphs = self.theme.glyphs
        width = self.layout.bar
        filled = width * (done / total if total > 0 else 1.0)
        full = min(width, int(filled))
        half = 1 if full < width and filled - full >= 0.5 else 0
        return [
            (glyphs.bar_full * full + glyphs.bar_half * half, (style,)),
            (glyphs.bar_empty * (width - full - half), (Style.DIM,)),
        ]

    @staticmethod
    def _percent(file: FileProgress) -> str:
        """Format the percent of a file.

        Args:
            file: The file.

        Returns:
            For example "42%".
        """
        return f"{percent_of(file.bytes_on_disk, file.size)}%"

    def _icon(self, state: FileState) -> tuple[str, Style]:
        """Choose the icon of a file state.

        Args:
            state: An active file state.

        Returns:
            The icon and its style. A file being checked gets a spinner frame.
        """
        glyphs = self.theme.glyphs
        if state is FileState.RETRYING:
            return glyphs.retrying, Style.YELLOW
        if state is FileState.VERIFYING:
            frame = int(self.now * SPINNER_FRAMES_PER_SECOND) % len(glyphs.spinner)
            return glyphs.spinner[frame], Style.CYAN
        return glyphs.downloading, Style.CYAN

    def _status(self, file: FileProgress) -> tuple[str, Style]:
        """Describe a file that retries or is being checked, in place of its bar.

        Args:
            file: A file in the `RETRYING` or `VERIFYING` state.

        Returns:
            The text, for example "retry 2 in 4 s · no data for 60 s", and its style.
        """
        glyphs = self.theme.glyphs
        if file.state is FileState.VERIFYING:
            return f"checking checksum{glyphs.ellipsis}", Style.DIM
        retry = file.retry
        if retry is None:
            return "retrying", Style.YELLOW
        wait = format_duration(max(0.0, retry.next_attempt_at - self.now))
        return f"retry {retry.attempt} in {wait} {glyphs.dot} {retry.reason}", Style.YELLOW
