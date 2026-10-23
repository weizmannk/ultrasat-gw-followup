"""Download the observing-scenario archive from Zenodo.

The record is pinned, not resolved through the concept DOI: the paper was
produced from one specific version of the data, and a later version would
change the results silently. An interrupted download is resumed, and the MD5
checksum published by Zenodo is verified before the file is accepted.

Usage
-----
::

    python -m ultrasat_gw_followup.workflow.download --config config/ultrasat_lcs.ini
    ugf-download --record 22550048 --file-name runs.zip --data-dir data

Output: ``<data_dir>/<file_name>``. Exits non-zero if anything fails.
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import requests
from tqdm.auto import tqdm

from ultrasat_gw_followup.config import Config

#: Zenodo record holding the GWTC-5.0 observing scenarios used by the paper.
DEFAULT_RECORD = "22550048"

ZENODO_API = "https://zenodo.org/api/records"
CHUNK_SIZE = 1 << 20

log = logging.getLogger(__name__)


def parse_arguments(argv=None):
    """Parse the command line, with the ``.ini`` values as defaults."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], allow_abbrev=False
    )
    config, remaining = Config.from_command_line(parser, argv)

    parser.add_argument(
        "--record",
        default=config.get("download", "record", DEFAULT_RECORD),
        help="pinned Zenodo record id (not the concept DOI)",
    )
    parser.add_argument(
        "--file-name",
        default=config.get("download", "file_name", "runs.zip"),
        help="name of the file to download from the record",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(config.get("paths", "data_dir", "data")),
        help="directory the archive is written to",
    )
    return parser.parse_args(remaining)


def fetch_file_metadata(record, file_name, session=None):
    """Return the Zenodo metadata of one file of a record.

    Parameters
    ----------
    record : str
        Pinned record id.
    file_name : str
        Name of the file inside the record.
    session : requests.Session, optional
        Session to issue the request with.

    Returns
    -------
    dict
        The entry of ``files`` whose ``key`` is ``file_name``.

    Raises
    ------
    requests.HTTPError
        If the record cannot be read.
    KeyError
        If the record holds no file of that name.
    """
    get = (session or requests).get
    response = get(f"{ZENODO_API}/{record}", timeout=60)
    response.raise_for_status()
    files = response.json().get("files", [])
    for entry in files:
        if entry["key"] == file_name:
            return entry
    raise KeyError(
        f"Zenodo record {record} has no file {file_name!r}; "
        f"it holds {sorted(entry['key'] for entry in files)}"
    )


def md5sum(path, chunk_size=CHUNK_SIZE):
    """Return the hex MD5 digest of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def expected_md5(metadata):
    """Return the MD5 digest Zenodo publishes for a file, without its prefix.

    Raises
    ------
    ValueError
        If the record publishes a checksum in some other algorithm, which
        this step cannot verify.
    """
    checksum = metadata.get("checksum", "")
    algorithm, _, digest = checksum.partition(":")
    if algorithm != "md5" or not digest:
        raise ValueError(f"record does not publish an MD5 checksum: {checksum!r}")
    return digest


def download(url, path, resume_from=None, session=None):
    """Download ``url`` to ``path``, resuming a partial file if there is one.

    Parameters
    ----------
    url : str
        Direct link to the file content.
    path : pathlib.Path
        Destination. An existing shorter file is continued, not overwritten.
    resume_from : int, optional
        Byte to continue from. Defaults to the size of ``path``; pass 0 to
        start over on a file that is complete but does not verify.
    session : requests.Session, optional
        Session to issue the request with.

    Raises
    ------
    requests.HTTPError
        If the transfer cannot be started.
    """
    get = (session or requests).get
    if resume_from is None:
        resume_from = path.stat().st_size if path.exists() else 0
    downloaded = resume_from
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

    response = get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    # A resume is only valid if the server answered 206 Partial Content.
    # On 200 it is sending the whole file, so appending would corrupt it.
    if downloaded and response.status_code != 206:
        log.warning("server ignored the range request; restarting the download")
        downloaded = 0

    total = int(response.headers.get("content-length", 0)) + downloaded
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(path, "ab" if downloaded else "wb") as stream,
        tqdm(
            total=total or None,
            initial=downloaded,
            unit="B",
            unit_scale=True,
            desc=path.name,
        ) as progress,
    ):
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            stream.write(chunk)
            progress.update(len(chunk))


def fetch(record, file_name, data_dir, session=None):
    """Download one file of a Zenodo record and verify its checksum.

    A file already on disk with the right digest is kept as is. A shorter one
    is resumed. One that is already the published size, or longer, but does
    not verify is downloaded again from the start: resuming it would ask the
    server for nothing and leave the same bad bytes in place.

    Parameters
    ----------
    record : str
        Pinned record id.
    file_name : str
        Name of the file inside the record.
    data_dir : str or pathlib.Path
        Directory to write to.
    session : requests.Session, optional
        Session to issue the requests with.

    Returns
    -------
    pathlib.Path
        Path of the verified file.

    Raises
    ------
    ValueError
        If the downloaded file does not match the published MD5 digest.
    """
    path = Path(data_dir) / file_name
    metadata = fetch_file_metadata(record, file_name, session)
    digest = expected_md5(metadata)
    size = metadata.get("size")

    resume_from = None
    if path.exists():
        local = path.stat().st_size
        # Only hash a file that is already the full length; on a 37 GiB
        # archive, hashing a partial one would cost minutes to learn nothing.
        if size is not None and local < size:
            log.info("%s is short (%d of %d bytes), resuming", path, local, size)
        elif md5sum(path) == digest:
            log.info("%s is already present and its MD5 matches", path)
            return path
        else:
            log.warning("%s does not match its MD5, downloading it again", path)
            resume_from = 0

    download(metadata["links"]["self"], path, resume_from, session)

    found = md5sum(path)
    if found != digest:
        raise ValueError(f"{path}: MD5 is {found}, expected {digest}")
    log.info("downloaded %s, MD5 %s", path, found)
    return path


def main(argv=None):
    """Entry point of ``ugf-download``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_arguments(argv)
    try:
        fetch(args.record, args.file_name, args.data_dir)
    except Exception as error:  # noqa: BLE001 -- any failure must exit non-zero
        log.error("download failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
