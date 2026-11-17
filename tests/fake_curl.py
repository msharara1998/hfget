"""A stand-in for curl, copied into a folder on PATH by `test_downloader.py`.

It reads `--output <file>` and the URL (the last argument) from its arguments. The URL ends
with a file id. The folder in the `FAKE_CURL_STATE` environment variable holds, per id:

- `<id>.bin`: the full content of the file.
- `<id>.json`: a list of steps, one per call. The last step repeats; no steps means "send
  the rest of the file".
- `<id>.calls`: one JSON line per call, written here: the arguments, what came on stdin,
  and the size of the output file before the call.

A step may set:

- `append`: how many bytes to add from where the output file ends (default: all the rest
  when the exit code is 0, else nothing);
- `corrupt`: add wrong bytes instead of the real ones;
- `exit`: the exit code (default 0);
- `http`: the HTTP code written on stdout (default 206 when resuming, else 200);
- `stderr`: the error text.
"""

import json
import os
import sys
from pathlib import Path


def main() -> int:
    args = sys.argv[1:]
    output = Path(args[args.index("--output") + 1])
    file_id = args[-1].rsplit("/", 1)[-1]
    state = Path(os.environ["FAKE_CURL_STATE"])
    content = (state / f"{file_id}.bin").read_bytes()
    steps = json.loads((state / f"{file_id}.json").read_text()) or [{}]
    log = state / f"{file_id}.calls"
    calls = len(log.read_text().splitlines()) if log.exists() else 0
    step = steps[min(calls, len(steps) - 1)]

    size_before = output.stat().st_size if output.exists() else 0
    record = {"argv": args, "stdin": sys.stdin.read(), "size_before": size_before}
    with log.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    exit_code = step.get("exit", 0)
    count = step.get("append", None if exit_code == 0 else 0)
    end = len(content) if count is None else size_before + count
    chunk = content[size_before:end]
    if step.get("corrupt"):
        chunk = bytes(byte ^ 0xFF for byte in chunk)
    with output.open("ab") as fh:
        fh.write(chunk)

    sys.stdout.write(step.get("http", "206" if size_before else "200"))
    sys.stderr.write(step.get("stderr", ""))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
