"""The Zenodo download: pinned record, resume, and MD5 verification.

No request reaches the network: a fake session answers from an in-memory
record, so the resume logic and the checksum check are exercised exactly as
they are against Zenodo.
"""

import hashlib
from pathlib import Path

import pytest
import requests

from ultrasat_gw_followup.workflow import download

CONFIGS = Path(__file__).resolve().parents[1] / "config"
CONTENT = b"observing scenarios archive" * 100
DIGEST = hashlib.md5(CONTENT).hexdigest()
RECORD = "22555949"
FILE_NAME = "runs.zip"


class FakeResponse:
    """The subset of ``requests.Response`` the downloader uses."""

    def __init__(self, status_code, payload=None, body=b"", headers=None):
        self.status_code = status_code
        self._payload = payload
        self._body = body
        self.headers = headers or {"content-length": str(len(body))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    def iter_content(self, chunk_size):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]


class FakeSession:
    """A Zenodo record holding one file, optionally refusing to resume."""

    def __init__(self, content=CONTENT, digest=DIGEST, honour_range=True):
        self.content = content
        self.digest = digest
        self.honour_range = honour_range
        self.ranges = []

    def get(self, url, headers=None, stream=False, timeout=None):
        if url.endswith(f"/{RECORD}"):
            return FakeResponse(
                200,
                payload={
                    "files": [
                        {
                            "key": FILE_NAME,
                            "size": len(self.content),
                            "checksum": f"md5:{self.digest}",
                            "links": {"self": "https://example.invalid/runs.zip"},
                        }
                    ]
                },
            )
        start = 0
        if headers and "Range" in headers:
            self.ranges.append(headers["Range"])
            if self.honour_range:
                start = int(headers["Range"].removeprefix("bytes=").rstrip("-"))
                return FakeResponse(206, body=self.content[start:])
        return FakeResponse(200, body=self.content[start:])


def test_the_record_is_pinned():
    assert download.DEFAULT_RECORD == RECORD


@pytest.mark.parametrize("name", ["ultrasat_lcs", "ultrasat_allsky", "uvex"])
def test_every_run_config_points_at_that_record(name):
    args = download.parse_arguments(["--config", str(CONFIGS / f"{name}.ini")])
    assert args.record == RECORD
    assert args.file_name == FILE_NAME


def test_download_and_verify(tmp_path):
    session = FakeSession()
    path = download.fetch(RECORD, FILE_NAME, tmp_path, session)
    assert path.read_bytes() == CONTENT
    assert download.md5sum(path) == DIGEST
    assert session.ranges == []


def test_an_interrupted_download_is_resumed(tmp_path):
    (tmp_path / FILE_NAME).write_bytes(CONTENT[:500])
    session = FakeSession()
    path = download.fetch(RECORD, FILE_NAME, tmp_path, session)
    assert session.ranges == ["bytes=500-"]
    assert path.read_bytes() == CONTENT


def test_a_server_ignoring_the_range_restarts_the_download(tmp_path):
    """Appending to a 200 response would corrupt the file."""
    (tmp_path / FILE_NAME).write_bytes(CONTENT[:500])
    session = FakeSession(honour_range=False)
    path = download.fetch(RECORD, FILE_NAME, tmp_path, session)
    assert path.read_bytes() == CONTENT


def test_a_complete_file_is_not_downloaded_again(tmp_path):
    (tmp_path / FILE_NAME).write_bytes(CONTENT)
    session = FakeSession()
    download.fetch(RECORD, FILE_NAME, tmp_path, session)
    assert session.ranges == []


def test_a_complete_but_corrupt_file_is_downloaded_from_the_start(tmp_path):
    """Resuming it would ask for nothing and leave the same bad bytes."""
    path = tmp_path / FILE_NAME
    path.write_bytes(b"x" * len(CONTENT))
    session = FakeSession()

    assert download.fetch(RECORD, FILE_NAME, tmp_path, session).read_bytes() == CONTENT
    assert session.ranges == []


def test_a_file_longer_than_published_is_downloaded_from_the_start(tmp_path):
    path = tmp_path / FILE_NAME
    path.write_bytes(CONTENT + b"trailing garbage")
    session = FakeSession()

    assert download.fetch(RECORD, FILE_NAME, tmp_path, session).read_bytes() == CONTENT
    assert session.ranges == []


def test_a_wrong_checksum_is_an_error(tmp_path):
    session = FakeSession(digest="0" * 32)
    with pytest.raises(ValueError, match="MD5"):
        download.fetch(RECORD, FILE_NAME, tmp_path, session)


def test_an_unknown_file_is_an_error(tmp_path):
    with pytest.raises(KeyError):
        download.fetch(RECORD, "absent.zip", tmp_path, FakeSession())


def test_a_record_without_an_md5_cannot_be_verified():
    with pytest.raises(ValueError, match="MD5"):
        download.expected_md5({"checksum": "sha256:abc"})


def test_main_exits_non_zero_on_failure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(download.requests, "get", fail)
    assert download.main(["--record", RECORD, "--data-dir", str(tmp_path)]) == 1
