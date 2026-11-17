"""How progress text looks: glyphs, colors, units, and widths in terminal cells.

Used only by the command line display. Every function here is pure.
"""

import os
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import astuple, dataclass
from enum import Enum
from typing import TextIO, TypeAlias

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
UNITS = ("kB", "MB", "GB", "TB")


class Style(str, Enum):
    """An ANSI text style. The value is its SGR code."""

    BOLD = "1"
    DIM = "2"
    RED = "31"
    GREEN = "32"
    YELLOW = "33"
    CYAN = "36"


Segment: TypeAlias = tuple[str, tuple[Style, ...]]
"""A piece of text and its styles. A line is a list of segments."""


@dataclass(frozen=True)
class Glyphs:
    """The characters that draw icons, bars and separators.

    Attributes:
        downloading: The icon of a file that downloads.
        done: The icon of a finished file.
        retrying: The icon of a file that waits before a retry.
        spinner: The frames of the spinner shown while a checksum is checked.
        bar_full: A filled bar cell.
        bar_half: A half-filled bar cell.
        bar_empty: An empty bar cell.
        ellipsis: Marks text that was cut.
        arrow: Points from the repo to the destination folder.
        dot: Separates items on a line.
    """

    downloading: str
    done: str
    retrying: str
    spinner: str
    bar_full: str
    bar_half: str
    bar_empty: str
    ellipsis: str
    arrow: str
    dot: str

    @classmethod
    def for_stream(cls, stream: TextIO) -> "Glyphs":
        """Choose Unicode glyphs if the stream can encode them, else ASCII ones.

        Args:
            stream: The stream the glyphs will be written to.

        Returns:
            `UNICODE_GLYPHS` or `ASCII_GLYPHS`.
        """
        try:
            "".join(astuple(UNICODE_GLYPHS)).encode(stream.encoding or "ascii")
        except (UnicodeEncodeError, LookupError):
            return cls(*astuple(ASCII_GLYPHS))
        return cls(*astuple(UNICODE_GLYPHS))


UNICODE_GLYPHS = Glyphs(
    downloading="↓",
    done="✓",
    retrying="↻",
    spinner="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
    bar_full="━",
    bar_half="╸",
    bar_empty="─",
    ellipsis="…",
    arrow="→",
    dot="·",
)
ASCII_GLYPHS = Glyphs(
    downloading=">",
    done="+",
    retrying="!",
    spinner="|/-\\",
    bar_full="#",
    bar_half=">",
    bar_empty="-",
    ellipsis="...",
    arrow="->",
    dot="-",
)


@dataclass(frozen=True)
class Theme:
    """Glyphs plus the choice to use colors.

    Attributes:
        glyphs: The characters to draw with.
        color: Whether to add ANSI color codes.
    """

    glyphs: Glyphs
    color: bool

    @classmethod
    def for_stream(cls, stream: TextIO) -> "Theme":
        """Choose glyphs for a stream, with colors unless `NO_COLOR` is set.

        Args:
            stream: The stream the display writes to.

        Returns:
            The theme.
        """
        return cls(Glyphs.for_stream(stream), color=not os.environ.get("NO_COLOR"))

    def render(self, segments: Sequence[Segment], width: int) -> str:
        """Join segments into one line that is at most `width` cells wide.

        Text past the width is cut and ends with the ellipsis glyph.

        Args:
            segments: The pieces of the line.
            width: The maximum visible width in terminal cells.

        Returns:
            The line, with color codes if the theme uses colors.
        """
        fitted = strip_trailing_spaces(fit_segments(segments, width, self.glyphs.ellipsis))
        return "".join(self.paint(text, *styles) for text, styles in fitted)

    def paint(self, text: str, *styles: Style) -> str:
        """Add color codes around a text.

        Args:
            text: The text.
            *styles: The styles to apply.

        Returns:
            The text with ANSI codes, or the plain text without colors or styles.
        """
        if not (self.color and styles and text):
            return text
        return f"\x1b[{';'.join(style.value for style in styles)}m{text}\x1b[0m"


def fit_segments(segments: Sequence[Segment], width: int, ellipsis: str) -> list[Segment]:
    """Cut a line of segments to a width, ending it with an ellipsis if it was cut.

    Args:
        segments: The pieces of the line.
        width: The maximum width in terminal cells.
        ellipsis: The text that marks the cut.

    Returns:
        Segments whose total width is at most `width`.
    """
    if sum(cell_width(text) for text, _ in segments) <= width:
        return list(segments)
    marker = ellipsis if cell_width(ellipsis) <= width else ""
    budget = width - cell_width(marker)
    fitted: list[Segment] = []
    styles: tuple[Style, ...] = ()
    for text, styles in segments:
        piece = take_prefix(text, budget)
        fitted.append((piece, styles))
        budget -= cell_width(piece)
        if piece != text:
            break
    fitted.append((marker, styles))
    return fitted


def strip_trailing_spaces(segments: Sequence[Segment]) -> list[Segment]:
    """Remove the spaces at the end of a line, such as the padding of its last column.

    Args:
        segments: The pieces of the line.

    Returns:
        The segments without trailing spaces or trailing empty segments.
    """
    stripped = list(segments)
    while stripped:
        text, styles = stripped[-1]
        if text.rstrip(" "):
            stripped[-1] = (text.rstrip(" "), styles)
            break
        stripped.pop()
    return stripped


def char_width(char: str) -> int:
    """Return how many terminal cells a character takes.

    Args:
        char: One character.

    Returns:
        2 for wide East Asian characters, 0 for combining and control characters, else 1.
    """
    if unicodedata.combining(char) or unicodedata.category(char) in ("Mn", "Me", "Cf", "Cc"):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def cell_width(text: str) -> int:
    """Return how many terminal cells a text takes.

    Args:
        text: Text without ANSI codes.

    Returns:
        The width in cells.
    """
    return sum(char_width(char) for char in text)


def visible_width(line: str) -> int:
    """Return how many terminal cells a line takes once its ANSI codes are removed.

    Args:
        line: A line that may contain ANSI codes.

    Returns:
        The width in cells.
    """
    return cell_width(ANSI_ESCAPE.sub("", line))


def take_prefix(text: str, width: int) -> str:
    """Return the longest start of a text that fits in a width.

    Args:
        text: The text.
        width: The width in cells.

    Returns:
        The start of the text.
    """
    used = 0
    for index, char in enumerate(text):
        used += char_width(char)
        if used > width:
            return text[:index]
    return text


def take_suffix(text: str, width: int) -> str:
    """Return the longest end of a text that fits in a width.

    Args:
        text: The text.
        width: The width in cells.

    Returns:
        The end of the text.
    """
    return take_prefix(text[::-1], width)[::-1]


def truncate_middle(text: str, width: int, ellipsis: str) -> str:
    """Shorten a text to a width by cutting its middle.

    More of the end is kept than of the start, so endings such as
    `00001-of-00002.safetensors` stay visible.

    Args:
        text: The text.
        width: The maximum width in cells.
        ellipsis: The text that replaces the cut part.

    Returns:
        The text, shortened if needed.
    """
    if cell_width(text) <= width:
        return text
    room = width - cell_width(ellipsis)
    if room < 2:
        return take_prefix(text, width)
    head = room // 4
    return take_prefix(text, head) + ellipsis + take_suffix(text, room - head)


def pad(text: str, width: int, *, right: bool = False) -> str:
    """Pad a text with spaces to a width in cells.

    Args:
        text: The text. It must not be wider than `width`.
        width: The width in cells.
        right: Align the text to the right instead of the left.

    Returns:
        The padded text.
    """
    fill = " " * max(0, width - cell_width(text))
    return fill + text if right else text + fill


def scale_bytes(n: float) -> tuple[float, str]:
    """Express a byte count in decimal units.

    Args:
        n: A number of bytes.

    Returns:
        The value and its unit, for example (3.98, "GB").
    """
    if n < 1000:
        return n, "B"
    value = float(n)
    for unit in UNITS:
        value /= 1000
        if value < 999.5:
            return value, unit
    return value, UNITS[-1]


def _number(value: float, unit: str, significant: int) -> str:
    """Format a scaled value with a number of significant digits.

    Args:
        value: The scaled value.
        unit: Its unit. Bytes ("B") are always whole numbers.
        significant: 2 or 3 significant digits.

    Returns:
        For example "3.98" (3 digits) or "4.0" (2 digits).
    """
    if unit == "B":
        return f"{value:.0f}"
    decimals = max(0, significant - len(str(int(value))))
    return f"{value:.{decimals}f}"


def format_bytes(n: float) -> str:
    """Format a size with three significant digits, for example "3.98 GB".

    Args:
        n: A number of bytes.

    Returns:
        The formatted size.
    """
    value, unit = scale_bytes(n)
    return f"{_number(value, unit, 3)} {unit}"


def format_byte_pair(done: float, total: float) -> str:
    """Format "done / total", sharing the unit when both use the same one.

    Args:
        done: The bytes done.
        total: The total bytes.

    Returns:
        For example "3.91 / 9.32 GB" or "640 MB / 1.50 GB".
    """
    done_value, done_unit = scale_bytes(done)
    total_value, unit = scale_bytes(total)
    if done_unit != unit:
        return f"{format_bytes(done)} / {format_bytes(total)}"
    return f"{_number(done_value, unit, 3)} / {_number(total_value, unit, 3)} {unit}"


def format_speed(bytes_per_second: float | None) -> str:
    """Format a speed with two significant digits, for example "2.1 MB/s".

    Args:
        bytes_per_second: The speed, or None if it is not known yet.

    Returns:
        The formatted speed, or "--" if it is not known.
    """
    if bytes_per_second is None:
        return "--"
    value, unit = scale_bytes(max(0.0, bytes_per_second))
    return f"{_number(value, unit, 2)} {unit}/s"


def format_duration(seconds: float) -> str:
    """Format a duration for people, for example "42 s", "3 min 05 s" or "1 h 05 min".

    Args:
        seconds: The duration. Zero or less shows as "0 s"; any other duration under one
            second shows as "1 s".

    Returns:
        The formatted duration.
    """
    if seconds <= 0:
        return "0 s"
    total = max(1, round(seconds))
    if total < 60:
        return f"{total} s"
    minutes, secs = divmod(total, 60)
    if minutes < 10:
        return f"{minutes} min {secs:02d} s"
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def plural(count: int, singular: str, many: str | None = None) -> str:
    """Write a count with a word in the right number, for example "1 file" or "2 files".

    Args:
        count: The count.
        singular: The word for one item.
        many: The word for several items. None means `singular` plus "s".

    Returns:
        The count and the word.
    """
    word = singular if count == 1 else (many or f"{singular}s")
    return f"{count} {word}"
