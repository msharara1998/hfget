import shutil
from pathlib import Path

import pytest

from hfget.curl import Curl, CurlResult
from hfget.errors import DownloadError

TOKEN = "hf_secret_token_123"


@pytest.fixture
def curl_installed(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")


def test_succeeded():
    assert CurlResult(0, "200", "").succeeded
    assert not CurlResult(56, "206", "").succeeded


@pytest.mark.parametrize("http_code", ["401", "403", "404"])
def test_fatal_http_errors(http_code):
    assert CurlResult(22, http_code, "").is_fatal_http_error


@pytest.mark.parametrize(
    ("exit_code", "http_code"),
    [(22, "500"), (22, "429"), (22, ""), (56, "404"), (0, "404")],
)
def test_not_fatal_http_errors(exit_code, http_code):
    assert not CurlResult(exit_code, http_code, "").is_fatal_http_error


def test_write_and_range_errors():
    assert CurlResult(23, "200", "").is_write_error
    assert not CurlResult(33, "", "").is_write_error
    assert CurlResult(33, "", "").is_range_error
    assert not CurlResult(23, "200", "").is_range_error


@pytest.mark.parametrize(
    ("exit_code", "http_code", "reason"),
    [
        (56, "206", "connection reset"),
        (28, "206", "no data for 60 s"),
        (33, "", "server cannot resume"),
        (22, "503", "HTTP 503"),
        (22, "", "curl exit code 22"),
        (99, "", "curl exit code 99"),
        (-15, "", "curl was stopped"),
    ],
)
def test_reason(exit_code, http_code, reason):
    assert CurlResult(exit_code, http_code, "").reason == reason


def test_token_goes_on_stdin_not_on_the_command_line(curl_installed):
    curl = Curl(TOKEN)
    cmd = curl._command("https://hub.test/file", Path("/tmp/file.incomplete"))
    assert all(TOKEN not in arg for arg in cmd)
    assert cmd[0] == "curl"
    assert cmd[-1] == "https://hub.test/file"
    assert cmd[cmd.index("--config") + 1] == "-"
    assert cmd[cmd.index("--continue-at") + 1] == "-"
    assert cmd[cmd.index("--output") + 1] == "/tmp/file.incomplete"
    assert cmd[cmd.index("--write-out") + 1] == "%{http_code}"
    assert curl._config() == f'header = "Authorization: Bearer {TOKEN}"\n'


def test_no_token_means_no_config(curl_installed):
    curl = Curl(None)
    assert "--config" not in curl._command("https://hub.test/file", Path("out"))
    assert curl._config() == ""


def test_missing_curl_raises(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(DownloadError, match="curl is not installed"):
        Curl(None)


def test_missing_curl_on_empty_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(DownloadError, match="not on PATH"):
        Curl(TOKEN)
