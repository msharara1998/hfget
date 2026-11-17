# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-17

First release.

### Added
- `hfget` command with the same arguments as `hf download`.
- Partial files continue from where they stopped, with `curl --continue-at -`: automatic
  retries after network errors, and a plain rerun after Ctrl+C, a closed terminal or a reboot.
- Checksum check of every finished file (sha256, or the git blob id for small files).
- Same disk layouts as `hf download`: `--local-dir` and the Hugging Face cache.
- Live progress display that stays clean when the terminal is resized, and plain progress
  lines when the output is not a terminal.
- Python API: `download`, `plan_download`, `execute_plan`, and progress snapshots through a
  callback.

[0.1.0]: https://github.com/msharara1998/hfget/releases/tag/v0.1.0
