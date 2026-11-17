"""The `hfget` command: a thin command line layer over the `api` module.

It takes the same arguments as `hf download`. It prints the final path on stdout, sends
messages to stderr, and turns errors into exit messages.

Usage:
    hfget Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
"""

import argparse
import json
import logging
import sys
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path

from . import __version__
from .api import DownloadPlan, execute_plan, plan_download
from .display import Display, LiveDisplay, PlainDisplay
from .errors import DownloadError, StoppedError
from .hub import REPO_TYPES
from .progress import is_progress_event
from .style import Glyphs, Theme
from .utils import human_size

OUTPUT_FORMATS = ("auto", "human", "agent", "json", "quiet")
VERBOSE_FORMATS = ("auto", "human", "agent")  # the formats that show info messages
STOPPED_MESSAGE = "\nStopped. Finished files are kept. Run the same command again to continue."


class _ConsoleHandler(logging.Handler):
    """Writes log messages to stderr, or through the progress display while one is shown.

    Attributes:
        display: The display that receives the messages, or None to write to stderr.
    """

    def __init__(self) -> None:
        """Create the handler with no display attached."""
        super().__init__()
        self.display: Display | None = None

    def emit(self, record: logging.LogRecord) -> None:
        """Write one log record.

        Args:
            record: The record to write.
        """
        try:
            message = self.format(record)
            display = self.display
            if display is None:
                sys.stderr.write(f"{message}\n")
                sys.stderr.flush()
            else:
                display.show_message(message, progress_event=is_progress_event(record))
        except Exception:  # the same handling as logging.StreamHandler
            self.handleError(record)

    @contextmanager
    def routed_to(self, display: Display) -> Generator[None, None, None]:
        """Send messages through a display while the `with` block runs.

        Args:
            display: The display.

        Yields:
            Nothing.
        """
        self.display = display
        try:
            yield
        finally:
            self.display = None


def main(argv: Sequence[str] | None = None) -> None:
    """Run the `hfget` command.

    Args:
        argv: The command line arguments. None means `sys.argv[1:]`.

    Raises:
        SystemExit: With an error message on stderr if the download fails or is stopped.
    """
    args = _parse_args(argv)
    output = _output_format(args)
    handler = _configure_logging(output)
    try:
        path = _run(args, output, handler)
    except DownloadError as e:
        sys.exit(f"Error: {e}")
    except (KeyboardInterrupt, StoppedError):
        sys.exit(STOPPED_MESSAGE)
    if args.dry_run:
        return  # `_run` already printed the plan
    if output == "json":
        print(json.dumps({"path": str(path)}))
    else:
        print(path)


def _run(args: argparse.Namespace, output: str, handler: _ConsoleHandler) -> Path:
    """Plan the download, then print the plan (dry run) or run it with a progress display.

    Args:
        args: The parsed command line.
        output: The output format.
        handler: The log handler, which is routed through the display while it is shown.

    Returns:
        The path to print: the requested file, or the folder.
    """
    plan = plan_download(
        args.repo_id,
        filenames=args.filenames,
        repo_type=args.repo_type,
        revision=args.revision,
        include=args.include,
        exclude=args.exclude,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
        force_download=args.force_download,
        token=args.token,
    )
    if args.dry_run:
        if output == "json":
            _print_dry_run_json(plan)
        else:
            _print_dry_run(plan)
        return plan.path
    display = _make_display(output, plan)
    if display is None:
        return execute_plan(plan, token=args.token, max_workers=args.max_workers)
    with closing(display), handler.routed_to(display):
        return execute_plan(
            plan, token=args.token, max_workers=args.max_workers, on_progress=display.update
        )


def _make_display(output: str, plan: DownloadPlan) -> Display | None:
    """Choose the progress display for an output format.

    Args:
        output: The output format.
        plan: The plan being run.

    Returns:
        A live display for `human`, and for `auto` on a terminal; plain progress lines for
        `agent`, and for `auto` when stderr is not a terminal; None for `json` and `quiet`.
    """
    if output == "human" or (output == "auto" and sys.stderr.isatty()):
        return LiveDisplay(sys.stderr, plan, Theme.for_stream(sys.stderr))
    if output in ("auto", "agent"):
        return PlainDisplay(sys.stderr, Glyphs.for_stream(sys.stderr))
    return None


def _print_dry_run(plan: DownloadPlan) -> None:
    """Print the files a download would fetch, in the format of `hf download --dry-run`.

    Args:
        plan: The download plan.
    """
    print(
        f"[dry-run] Will download {len(plan.pending)} files (out of {len(plan.files)}) "
        f"totalling {human_size(plan.bytes_to_download)}."
    )
    if plan.pending:
        print("file\tsize")
        for file in plan.pending:
            print(f"{file.path}\t{human_size(file.size)}")


def _print_dry_run_json(plan: DownloadPlan) -> None:
    """Print the plan of a dry run as one JSON object, so the output stays valid JSON.

    Args:
        plan: The download plan.
    """
    print(
        json.dumps(
            {
                "path": str(plan.path),
                "dry_run": True,
                "bytes_to_download": plan.bytes_to_download,
                "files": [{"path": file.path, "size": file.size} for file in plan.pending],
            }
        )
    )


def _output_format(args: argparse.Namespace) -> str:
    """Resolve the output format from `--format`, `--json` and `--quiet`.

    Args:
        args: The parsed command line.

    Returns:
        One of `OUTPUT_FORMATS`.
    """
    if args.json:
        return "json"
    if args.quiet:
        return "quiet"
    return args.format


def _configure_logging(output: str) -> _ConsoleHandler:
    """Send the package's log messages to stderr.

    Info messages are shown only in the verbose formats. Warnings are always shown.

    Args:
        output: The output format.

    Returns:
        The handler, so a progress display can take over its output.
    """
    handler = _ConsoleHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    package_logger = logging.getLogger(__package__)
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO if output in VERBOSE_FORMATS else logging.WARNING)
    package_logger.propagate = False
    return handler


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line. The arguments and help texts match `hf download`.

    Args:
        argv: The command line arguments. None means `sys.argv[1:]`.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(
        prog="hfget",
        description="Download files from the Hub. Same arguments as `hf download`, "
        "but partial files continue where they stopped.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "repo_id",
        help="The ID of the repo (e.g. `username/repo-name` or `spaces/username/repo-name`).",
    )
    p.add_argument(
        "filenames",
        nargs="*",
        help="Files to download (e.g. `config.json`, `data/metadata.jsonl`).",
    )
    p.add_argument(
        "--type",
        "--repo-type",
        dest="repo_type",
        choices=REPO_TYPES,
        default="model",
        help="The type of repository (model, dataset, space, or kernel).",
    )
    p.add_argument(
        "--revision", help="Git revision id which can be a branch name, a tag, or a commit hash."
    )
    p.add_argument(
        "--include",
        action="append",
        help="Glob patterns to include from files to download. eg: *.json",
    )
    p.add_argument(
        "--exclude", action="append", help="Glob patterns to exclude from files to download."
    )
    p.add_argument("--cache-dir", help="Directory where to save files.")
    p.add_argument(
        "--local-dir", help="If set, the downloaded file will be placed under this directory."
    )
    p.add_argument(
        "--force-download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, the files will be downloaded even if they are already cached.",
    )
    p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, perform a dry run without actually downloading the file.",
    )
    p.add_argument(
        "--token", help="A User Access Token generated from https://huggingface.co/settings/tokens."
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of workers to use for downloading files. Default is 8.",
    )
    fmt = p.add_argument_group("Formatting options")
    fmt.add_argument(
        "--format",
        choices=OUTPUT_FORMATS,
        default="auto",
        help="Output format. Defaults to 'auto'.",
    )
    fmt.add_argument(
        "--json", action="store_true", help="JSON output. Equivalent to '--format json'."
    )
    fmt.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet output. Equivalent to '--format quiet'."
    )
    fmt.add_argument(
        "--no-truncate", action="store_true", help="Accepted for compatibility. Has no effect."
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
