"""Progress displays for the command line: a live terminal view and plain log lines.

Used only by the `cli` module. The lines themselves come from the pure functions in
the `frames` module; this module only writes them and moves the cursor.
"""

import atexit
import math
import os
import shutil
import threading
from collections.abc import Sequence
from typing import Protocol, TextIO

from .api import DownloadPlan
from .frames import (
    build_done_line,
    build_frame,
    build_header,
    build_plain_line,
    build_summary,
    percent_of,
)
from .progress import DownloadProgress, FileState
from .speed import Stats, StatsTracker
from .style import Glyphs, Theme, visible_width

HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
CLEAR_LINE = "\x1b[2K"
CLEAR_BELOW = "\x1b[J"
BEGIN_SYNC = "\x1b[?2026h"  # terminals that support it draw the frame in one go
END_SYNC = "\x1b[?2026l"
PLAIN_PERCENT_STEP = 10
PLAIN_MAX_SILENCE_SECONDS = 30.0


class Display(Protocol):
    """What the command line needs from a progress display."""

    def update(self, snapshot: DownloadProgress) -> None:
        """Show a new progress snapshot.

        Args:
            snapshot: The newest snapshot.
        """

    def show_message(self, message: str, *, progress_event: bool) -> None:
        """Show a log message.

        Args:
            message: The formatted message.
            progress_event: Whether the display already shows this event itself.
        """

    def close(self) -> None:
        """Stop the display and leave the terminal clean."""


def terminal_size(stream: TextIO) -> os.terminal_size:
    """Read the current size of the terminal behind a stream.

    Args:
        stream: The stream the display writes to.

    Returns:
        The size, or the `shutil.get_terminal_size` fallback if the stream is no terminal.
    """
    try:
        size = os.get_terminal_size(stream.fileno())
    except (AttributeError, OSError, ValueError):
        return shutil.get_terminal_size()
    if size.columns <= 0 or size.lines <= 0:
        return shutil.get_terminal_size()
    return size


def rows_occupied(widths: Sequence[int], columns: int) -> int:
    """Count the screen rows that lines take after the terminal rewraps them.

    When a window gets narrower, many terminals rewrap lines that are already on screen,
    so a line wider than the new width takes several rows.

    Args:
        widths: The visible width of each line.
        columns: The current terminal width.

    Returns:
        The number of rows.
    """
    columns = max(1, columns)
    return sum(max(1, math.ceil(width / columns)) for width in widths)


class LiveDisplay:
    """A live area that redraws in place, with permanent lines printed above it.

    The frame is written with a newline after each line, so the cursor rests at the start
    of the line below it. Each redraw moves up by the rows the old frame takes at the
    current width (more rows than were drawn if the terminal rewrapped it), overwrites
    them and clears the rest of the screen. Every line is at most `width - 1` cells wide,
    so the terminal never wraps it by itself. All methods are thread-safe.
    """

    def __init__(self, stream: TextIO, plan: DownloadPlan, theme: Theme) -> None:
        """Create the display. Nothing is drawn before the first snapshot.

        Args:
            stream: The terminal stream (stderr).
            plan: The plan being run, for the header.
            theme: The glyphs and colors.
        """
        self._stream = stream
        self._plan = plan
        self._theme = theme
        self._stats = StatsTracker()
        self._lock = threading.Lock()
        self._snapshot: DownloadProgress | None = None
        self._last_stats: Stats | None = None
        self._frame_widths: list[int] = []  # visible widths of the frame on screen
        self._reported: set[str] = set()  # files already printed as done
        self._cursor_hidden = False
        self._closed = False

    def update(self, snapshot: DownloadProgress) -> None:
        """Print newly finished files above the live area and redraw it.

        The header is printed with the first snapshot.

        Args:
            snapshot: The newest snapshot.
        """
        with self._lock:
            if self._closed:
                return
            width = terminal_size(self._stream).columns
            permanent: list[str] = []
            if self._snapshot is None:
                permanent += build_header(self._plan, snapshot, width, self._theme)
                self._hide_cursor()
            for file in snapshot.files:
                if file.state is FileState.DONE and file.path not in self._reported:
                    self._reported.add(file.path)
                    permanent.append(build_done_line(file, width, self._theme))
            self._snapshot = snapshot
            self._last_stats = self._stats.add(snapshot)
            self._redraw(permanent)

    def show_message(self, message: str, *, progress_event: bool) -> None:
        """Print a log message above the live area.

        Progress events (Done, Checking the checksum, download interrupted) are dropped,
        because the live area shows them.

        Args:
            message: The formatted message.
            progress_event: Whether the live area already shows this event.
        """
        if progress_event:
            return
        with self._lock:
            if self._closed:
                self._write(f"{message}\n")
            else:
                self._redraw([message])

    def close(self) -> None:
        """Stop the display and show the cursor again.

        If every file is done, the live area is replaced by a summary line. Otherwise
        (Ctrl+C or an error) the last frame is redrawn and left on screen.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            snapshot, stats = self._snapshot, self._last_stats
            if snapshot is None or stats is None:
                return  # nothing was drawn
            if _all_done(snapshot):
                width = terminal_size(self._stream).columns
                self._draw([build_summary(snapshot, stats, width, self._theme)], [], width)
            else:
                self._redraw([])
            self._show_cursor()

    def _redraw(self, permanent: list[str]) -> None:
        """Print permanent lines, then draw the live area for the current terminal size.

        Must be called with the lock held.

        Args:
            permanent: Lines to print above the live area. They stay on screen.
        """
        size = terminal_size(self._stream)
        frame: list[str] = []
        if self._snapshot is not None and self._last_stats is not None:
            frame = build_frame(self._snapshot, self._last_stats, size, self._theme)
        self._draw(permanent, frame, size.columns)

    def _draw(self, permanent: list[str], frame: list[str], columns: int) -> None:
        """Erase the old frame, print permanent lines and draw a new frame in one write.

        Each row is cleared completely before its new line is written. A full clear also
        resets the "wrapped" mark that reflowing terminals such as tmux keep on a row, so
        the terminal never joins two of our lines when the window widens again. Only then
        is the rest of the screen cleared. Clearing from the top-left corner instead would
        make tmux copy the whole screen into its history (`scroll-on-clear`), which happens
        when a big shrink has pushed everything above the frame off screen.

        Must be called with the lock held.

        Args:
            permanent: Lines to print above the frame.
            frame: The new frame. An empty list leaves no live area.
            columns: The current terminal width.
        """
        rows = rows_occupied(self._frame_widths, columns)
        move_up = f"\r\x1b[{rows}A" if rows else ""
        body = "".join(f"{CLEAR_LINE}{line}\n" for line in [*permanent, *frame])
        self._write(f"{BEGIN_SYNC}{move_up}{body}{CLEAR_BELOW}{END_SYNC}")
        self._frame_widths = [visible_width(line) for line in frame]

    def _hide_cursor(self) -> None:
        """Hide the cursor, and make sure it comes back when the program exits."""
        if not self._cursor_hidden:
            self._cursor_hidden = True
            atexit.register(self._show_cursor)
            self._write(HIDE_CURSOR)

    def _show_cursor(self) -> None:
        """Show the cursor if this display hid it. Safe to call more than once."""
        if self._cursor_hidden:
            self._cursor_hidden = False
            self._write(SHOW_CURSOR)

    def _write(self, text: str) -> None:
        """Write to the terminal, ignoring a closed or broken stream.

        Args:
            text: The text and control codes to write.
        """
        try:
            self._stream.write(text)
            self._stream.flush()
        except (OSError, ValueError):
            pass


class PlainDisplay:
    """Plain progress lines, about every 10 % or every 30 s. No colors, no cursor moves."""

    def __init__(self, stream: TextIO, glyphs: Glyphs) -> None:
        """Create the display.

        Args:
            stream: The stream to write to (stderr).
            glyphs: The glyphs (only the separator is used).
        """
        self._stream = stream
        self._glyphs = glyphs
        self._stats = StatsTracker()
        self._lock = threading.Lock()
        self._last_step: int | None = None
        self._last_line_at = 0.0

    def update(self, snapshot: DownloadProgress) -> None:
        """Print a progress line when the next 10 % step is reached or 30 s have passed.

        Args:
            snapshot: The newest snapshot.
        """
        with self._lock:
            stats = self._stats.add(snapshot)
            step = percent_of(snapshot.done_bytes, snapshot.total_bytes) // PLAIN_PERCENT_STEP
            if self._last_step is None:  # the starting point of a resumed download
                self._last_step, self._last_line_at = step, snapshot.timestamp
                return
            silence = snapshot.timestamp - self._last_line_at
            if step > self._last_step or silence >= PLAIN_MAX_SILENCE_SECONDS:
                self._last_step, self._last_line_at = step, snapshot.timestamp
                self._write(build_plain_line(snapshot, stats, self._glyphs))

    def show_message(self, message: str, *, progress_event: bool) -> None:
        """Print a log message as it is, progress events included.

        Args:
            message: The formatted message.
            progress_event: Not used: plain output keeps every message.
        """
        with self._lock:
            self._write(message)

    def close(self) -> None:
        """Do nothing: plain output leaves nothing to clean up."""

    def _write(self, line: str) -> None:
        """Write one line.

        Args:
            line: The line, without a newline.
        """
        self._stream.write(f"{line}\n")
        self._stream.flush()


def _all_done(snapshot: DownloadProgress) -> bool:
    """Tell whether every file of a snapshot is done.

    Args:
        snapshot: A snapshot.

    Returns:
        True if all files are in the `DONE` state.
    """
    return all(file.state is FileState.DONE for file in snapshot.files)
