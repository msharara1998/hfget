# hfget

[![PyPI](https://img.shields.io/pypi/v/hfget)](https://pypi.org/project/hfget/)
[![Python versions](https://img.shields.io/pypi/pyversions/hfget)](https://pypi.org/project/hfget/)
[![License: MIT](https://img.shields.io/pypi/l/hfget)](https://github.com/msharara1998/hfget/blob/main/LICENSE)
[![CI](https://github.com/msharara1998/hfget/actions/workflows/ci.yml/badge.svg)](https://github.com/msharara1998/hfget/actions/workflows/ci.yml)

Download models and datasets from the Hugging Face Hub, and **continue partial files from
where they stopped**.

`hfget` takes the same arguments as `hf download` and saves files in the same layout, so
`transformers` and the other Hugging Face tools find them. The difference: when a download is
interrupted, `hfget` keeps the partial file and continues it, byte for byte.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/msharara1998/hfget/main/docs/why-hfget-dark.svg">
  <img alt="Diagram: a 5.3 GB file stops at 3.8 GB. hf download deletes the partial file, starts again at 0 GB and downloads 9.1 GB in total. hfget keeps the partial file, continues at 3.8 GB, downloads 5.3 GB in total and checks the checksum. While downloading, hfget shows live progress per file and hf download shows tqdm progress bars." src="https://raw.githubusercontent.com/msharara1998/hfget/main/docs/why-hfget-light.svg" width="100%">
</picture>

## Why hfget

Both tools skip files that already finished. The difference is what happens to a file that stops before the end:

| | `hf download` (huggingface_hub 2.4) | `hfget` |
| --- | --- | --- |
| The partial file | Deleted. Each run writes to a new temporary file and removes it when the download fails ([source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/file_download.py#L2004-L2046)). | Kept, as `<file>.incomplete`. |
| The next attempt | Starts that file again at 0 GB. | Continues from the last byte. curl asks the server only for the missing bytes. |
| Example: a 5.3 GB file stops at 3.8 GB | 9.1 GB downloaded in total | 5.3 GB downloaded in total |
| Progress display | tqdm progress bars | A live area with one row per file (bar, speed, ETA) and a total row. It redraws cleanly when you resize the terminal. |

On a slow or unstable connection, this can save hours on a large model. `hfget` also:

- **Continues by itself.** When the network drops, or no data arrives for 60 seconds, it
  waits and continues. The wait doubles after each failed attempt, up to 60 seconds.
- **Keeps partial files when you stop it.** After Ctrl+C, a closed terminal or a restart, run
  the same command again to continue.
- **Stops only when it should.** It stops after 10 attempts in a row bring no new data, and at
  once when retrying cannot help, for example when a repo is gated, private or not found.
- **Checks every file.** When a file is complete, its checksum is compared with the one from
  the Hub. A file that does not match is downloaded again.
- **Changes nothing else.** Same arguments as `hf download`, same folders and cache, so you can
  mix both tools. It also has a Python API with a progress callback.

## Install

```bash
pip install hfget
```

Or as a command-line tool with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install hfget     # installs the `hfget` command
uvx hfget --help          # or run it once without installing
```

Requirements: Python 3.10 or newer, and `curl` on your `PATH` (it does the downloading).
`curl` is preinstalled on macOS and most Linux systems. Otherwise:
`sudo apt install curl` (Debian/Ubuntu) or `brew install curl` (macOS).

## Quick start

```bash
# Download a model into a folder
hfget Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

# Or into the Hugging Face cache, where transformers looks for models
hfget Qwen/Qwen3-0.6B

# See what would be downloaded, without downloading
hfget Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B --dry-run

# Only some files
hfget Qwen/Qwen3-0.6B config.json tokenizer.json --local-dir models/Qwen3-0.6B
hfget Qwen/Qwen3-0.6B --include "*.json" --local-dir models/Qwen3-0.6B
```

If the download stops, run the same command again. To check a folder later, also run the same
command again: it prints `All files are already downloaded.` or downloads only what is
missing.

## What you see while downloading

In a terminal, two lines say what is downloaded and where. Below them, a live area shows one
row per active file and a total row:

```
Qwen/Qwen3.5-4B @ 851bf6e8 → models/Qwen3.5-4B
2 files to download (2 resuming, 3.98 GB already on disk) · 12 already complete
✓ config.json  3.20 kB
↓ model.safetensors-00001-of-00002.safetensors  ━━━━━━━━━━━━━━╸─────   71%  3.80 / 5.30 GB  1.9 MB/s
↻ model.safetensors-00002-of-00002.safetensors  retry 2 in 4 s · no data for 60 s
  Total 1/2                                      ━━━━━━━━━━━━━╸──────   65%  6.10 / 9.32 GB  2.1 MB/s  ETA 26 min  elapsed 41 min
```

- `↓` downloads, `↻` waits before a retry, and a spinner means the checksum is being checked.
- A finished file leaves the live area and stays above it as a `✓` line.
- Speeds cover the last 10 seconds and leave out bytes from earlier runs. ETA is the bytes
  left divided by the speed.
- When the window gets narrower, the speed, then the byte counts, then the bar are dropped.
  Long names are shortened in the middle, so the ending stays visible. The live area redraws
  correctly after the terminal rewraps lines when you resize it.
- At the end, a summary line replaces the live area, for example
  `✓ Downloaded 2 files (9.32 GB) in 1 h 12 min · average 2.1 MB/s`.
- Colors follow `NO_COLOR`. Plain ASCII characters are used if the terminal cannot show the
  Unicode ones.

When stderr is not a terminal (a log file or a pipe), or with `--format agent`, you get plain
lines instead, about every 10 % or 30 seconds:

```
[ 42%] 3.91 / 9.32 GB · 2.1 MB/s · ETA 43 min
```

`--quiet` and `--json` show no progress.

## Use it from Python

```python
from hfget import download

path = download("Qwen/Qwen3-0.6B", local_dir="models/Qwen3-0.6B")
print(path)  # models/Qwen3-0.6B
```

`download` takes the same options as the command line: `filenames`, `repo_type`, `revision`,
`include`, `exclude`, `cache_dir`, `local_dir`, `force_download`, `token`, `max_workers` and
`on_progress`. It returns the folder, or the file if you asked for exactly one file by name.

To look before you download, split it in two steps:

```python
from hfget import execute_plan, plan_download

plan = plan_download("Qwen/Qwen3-0.6B", local_dir="models/Qwen3-0.6B")
print(f"{len(plan.pending)} of {len(plan.files)} files, {plan.bytes_to_download} bytes to download")
execute_plan(plan)
```

To follow the progress, pass a callback. It receives a `DownloadProgress` snapshot about four
times a second:

```python
from hfget import DownloadProgress, download


def show(progress: DownloadProgress) -> None:
    print(f"{progress.done_bytes} / {progress.total_bytes} bytes")
    for file in progress.files:
        print(f"  {file.path}: {file.state.name} {file.bytes_on_disk} / {file.size}")


download("Qwen/Qwen3-0.6B", filenames=["config.json"], local_dir="models/qwen", on_progress=show)
```

The library never prints. It logs through the standard `logging` module under the `hfget`
logger, and raises `hfget.DownloadError` (a problem retrying will not fix) or
`hfget.StoppedError` (the download was stopped). Partial files are always kept, so calling
`download` again continues them.

## Command-line options

The options match `hf download`:

| Option | Meaning |
| --- | --- |
| `REPO_ID` | The repo, for example `Qwen/Qwen3-0.6B`. `datasets/...`, `spaces/...` and `hf://...` forms also work. |
| `FILENAMES...` | Only these files. A name that ends with `/` means a whole folder. |
| `--type`, `--repo-type` | `model` (default), `dataset`, `space` or `kernel` |
| `--revision` | A branch, tag or commit. Default: `main`. |
| `--include` / `--exclude` | Glob patterns, for example `--include "*.json"`. Repeat the option for more patterns. |
| `--local-dir` | Save the files in this folder. |
| `--cache-dir` | Hugging Face cache folder, used when `--local-dir` is not set. Default: `~/.cache/huggingface/hub`. |
| `--force-download` | Download files again even if they are already complete. Partial files are still continued. |
| `--dry-run` | Only show what would be downloaded. |
| `--token` | Hugging Face access token. Default: your saved login (`hf auth login`) or `HF_TOKEN`. |
| `--max-workers` | Number of files downloaded at the same time. Default: 8. |
| `--format`, `--json`, `-q` / `--quiet` | Output format. `human` shows the live progress area, `agent` prints plain progress lines, `auto` (default) picks `human` in a terminal and `agent` otherwise. |
| `--no-truncate` | Accepted so `hf download` commands work unchanged. It has no effect. |
| `--version` | Print the version. |

## How it works

1. **Find the files.** The `huggingface_hub` library gets the commit of the requested revision
   and lists every file with its size and checksum.
2. **Choose the files.** File names and `--include` / `--exclude` patterns are matched with the
   library's own matching function, so the selection is the same as `hf download`.
3. **Skip finished files.** A file is skipped when it already exists with the right size and
   checksum record.
4. **Download.** `curl --continue-at -` downloads each file into an `.incomplete` file. After
   an interruption, curl asks the server only for the bytes that are still missing.
5. **Check.** When a file is complete, its checksum is compared with the one from the Hub
   (sha256 for large files, the git blob id for small ones). A file that does not match is
   downloaded again.
6. **Move into place.** The file is moved to its final name and its record is written.

Your access token is passed to curl on stdin, so it never shows in the process list.

## Where the files go

With `--local-dir models/Qwen3-0.6B`:

```
models/Qwen3-0.6B/
├── config.json, tokenizer.json, ...      finished files
└── .cache/huggingface/download/
    ├── <file>.metadata                   record: commit, checksum, time
    └── <file>.incomplete                 partial file while downloading
```

Without `--local-dir`, files go to the Hugging Face cache (`blobs/`, `snapshots/`, `refs/`).
Both layouts are the same as `hf download` writes them, so you can mix the two tools.

## Platform support

- Tested in CI on Linux with Python 3.10 to 3.14, and on macOS with Python 3.10 and 3.14.
- Windows is not tested yet. The cache layout uses symlinks, which need Developer Mode on
  Windows.
- The live display is built for terminals that rewrap lines when you resize them, as most
  modern terminals do. It is tested in tmux. In terminals that cut long lines instead, such as
  classic xterm, shrinking the window may erase a few lines above the live area.

## Development

```bash
git clone https://github.com/msharara1998/hfget
cd hfget
uv sync
uv run ruff format && uv run ruff check && uv run ty check
uv run pytest               # offline tests
uv run pytest -m network    # also downloads a small real file from the Hub
```

## License

MIT. See [LICENSE](https://github.com/msharara1998/hfget/blob/main/LICENSE).
