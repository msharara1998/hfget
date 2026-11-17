import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / ".github" / "scripts" / "release_notes.py"
spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
assert spec is not None
assert spec.loader is not None
release_notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes)

CHANGELOG = """# Changelog

Intro text.

## [0.2.0] - 2027-01-10

### Fixed
- Second release.

## [0.1.0] - 2026-11-17

First release.

## [0.0.9] - 2026-10-01

[0.2.0]: https://example.com/v0.2.0
[0.1.0]: https://example.com/v0.1.0
"""


def test_section_stops_at_the_next_version():
    assert release_notes.changelog_section(CHANGELOG, "0.2.0") == "### Fixed\n- Second release."


def test_last_section_stops_at_the_link_definitions():
    changelog = CHANGELOG.replace("## [0.0.9] - 2026-10-01\n\n", "")
    assert release_notes.changelog_section(changelog, "0.1.0") == "First release."


def test_missing_version_is_an_error():
    with pytest.raises(ValueError, match=r"no section for version 3\.0\.0"):
        release_notes.changelog_section(CHANGELOG, "3.0.0")


def test_empty_section_is_an_error():
    with pytest.raises(ValueError, match="is empty"):
        release_notes.changelog_section(CHANGELOG, "0.0.9")


def test_version_dots_are_not_wildcards():
    with pytest.raises(ValueError, match="no section"):
        release_notes.changelog_section(CHANGELOG, "0x1x0")


def test_notes_add_install_lines_and_links():
    notes = release_notes.release_notes(CHANGELOG, "0.1.0")
    assert notes.startswith("First release.\n")
    assert "pip install hfget==0.1.0" in notes
    assert "https://pypi.org/project/hfget/0.1.0/" in notes
    assert "/blob/v0.1.0/CHANGELOG.md" in notes
