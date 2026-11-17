import json
import logging

import pytest

from hfget import cli
from hfget.api import DownloadPlan
from hfget.errors import DownloadError


@pytest.fixture(autouse=True)
def restore_package_logger():
    # `main` adds a handler to the `hfget` logger and stops propagation. Undo that, so the
    # other test files can still capture log messages.
    logger = logging.getLogger("hfget")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture
def files(remote_file):
    return (
        remote_file("config.json", b"x" * 1570),
        remote_file("model.safetensors", b"y" * 2_500_000),
        remote_file("README.md", b"z" * 10),
    )


def use_fixed_plan(monkeypatch, layout, files, pending):
    """Make `cli.plan_download` return a fixed plan, so the Hub is never called."""
    plan = DownloadPlan(
        repo_id="org/name",
        repo_type="model",
        commit=layout.commit,
        files=files,
        pending=pending,
        bytes_to_download=sum(file.size for file in pending),
        path=layout.root / "config.json",
        layout=layout,
    )
    calls = []

    def fake_plan_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return plan

    monkeypatch.setattr(cli, "plan_download", fake_plan_download)
    return plan, calls


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == "hfget 0.1.0\n"


def test_dry_run(capsys, monkeypatch, local_layout, files):
    _, calls = use_fixed_plan(monkeypatch, local_layout, files, pending=files[:2])

    cli.main(["datasets/org/name", "--dry-run", "--include", "*.json", "--revision", "v1"])

    assert capsys.readouterr().out.splitlines() == [
        "[dry-run] Will download 2 files (out of 3) totalling 2.5M.",
        "file\tsize",
        "config.json\t1.6K",
        "model.safetensors\t2.5M",
    ]
    [(repo_id, kwargs)] = calls
    assert repo_id == "datasets/org/name"
    assert kwargs["include"] == ["*.json"]
    assert kwargs["revision"] == "v1"


def test_dry_run_with_nothing_to_download(capsys, monkeypatch, local_layout, files):
    use_fixed_plan(monkeypatch, local_layout, files, pending=())
    cli.main(["org/name", "--dry-run"])
    assert capsys.readouterr().out == "[dry-run] Will download 0 files (out of 3) totalling 0.0.\n"


def test_json_prints_only_the_path(capsys, monkeypatch, local_layout, files):
    # Every file is already done, so the real `execute_plan` downloads nothing.
    plan, _ = use_fixed_plan(monkeypatch, local_layout, files, pending=())

    cli.main(["org/name", "config.json", "--json"])

    out = capsys.readouterr().out
    assert out == json.dumps({"path": str(plan.path)}) + "\n"
    assert json.loads(out) == {"path": str(plan.path)}


def test_json_dry_run_prints_one_json_object(capsys, monkeypatch, local_layout, files):
    plan, _ = use_fixed_plan(monkeypatch, local_layout, files, pending=files[:2])

    cli.main(["org/name", "--dry-run", "--json"])

    out = capsys.readouterr().out
    assert json.loads(out) == {
        "path": str(plan.path),
        "dry_run": True,
        "bytes_to_download": plan.bytes_to_download,
        "files": [{"path": file.path, "size": file.size} for file in files[:2]],
    }


def test_download_error_exits_with_message(monkeypatch):
    def failing_plan_download(repo_id, **kwargs):
        raise DownloadError("Repo not found: org/name")

    monkeypatch.setattr(cli, "plan_download", failing_plan_download)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["org/name"])
    assert exit_info.value.code == "Error: Repo not found: org/name"
