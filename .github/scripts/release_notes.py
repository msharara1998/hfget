"""Write the GitHub release notes of one version, taken from CHANGELOG.md.

The release workflow runs this before anything is published, so a version without a
changelog section stops the release early.

Usage:
    python3 .github/scripts/release_notes.py 0.1.0 > release-notes.md
"""

import argparse
import re
import sys
from pathlib import Path

PACKAGE = "hfget"
REPOSITORY = "msharara1998/hfget"


def changelog_section(changelog: str, version: str) -> str:
    """Return the body of one version's section in a Keep a Changelog file.

    The section starts after its `## [VERSION]` heading and ends at the next `## [` heading
    or at the link definitions at the end of the file.

    Args:
        changelog: The text of CHANGELOG.md.
        version: The version, for example "0.1.0".

    Returns:
        The section body without its heading, stripped of surrounding blank lines.

    Raises:
        ValueError: If the changelog has no section for the version, or the section is empty.
    """
    heading = re.compile(rf"^## \[{re.escape(version)}\][^\n]*\n", re.MULTILINE)
    start = heading.search(changelog)
    if start is None:
        raise ValueError(f"CHANGELOG.md has no section for version {version}.")
    end = re.compile(r"^(## \[|\[[^\]]+\]: )", re.MULTILINE).search(changelog, start.end())
    body = changelog[start.end() : end.start() if end else len(changelog)].strip()
    if not body:
        raise ValueError(f"The CHANGELOG.md section for version {version} is empty.")
    return body


def release_notes(changelog: str, version: str) -> str:
    """Build the release notes: the changelog section, install lines and links.

    Args:
        changelog: The text of CHANGELOG.md.
        version: The version, for example "0.1.0".

    Returns:
        The release notes in Markdown.
    """
    return f"""{changelog_section(changelog, version)}

### Install

```bash
pip install {PACKAGE}=={version}
```

or as a command-line tool: `uv tool install {PACKAGE}`, or run it once with `uvx {PACKAGE} --help`.

PyPI: https://pypi.org/project/{PACKAGE}/{version}/
Full changelog: https://github.com/{REPOSITORY}/blob/v{version}/CHANGELOG.md
"""


def main() -> None:
    """Print the release notes of the version given on the command line.

    Raises:
        SystemExit: With an error message if the changelog has no usable section.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="The version, for example 0.1.0 (without the v).")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    args = parser.parse_args()
    try:
        notes = release_notes(args.changelog.read_text(encoding="utf-8"), args.version)
    except ValueError as e:
        sys.exit(f"Error: {e}")
    sys.stdout.write(notes)


if __name__ == "__main__":
    main()
