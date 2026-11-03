"""Download the observing-scenario archive and extract events and sky maps.

The archive is fetched from a pinned Zenodo record, not through the concept
DOI, so that a later version of the data cannot change the results silently.
An interrupted download is resumed, and the MD5 checksum published by Zenodo
is verified before the file is accepted.

For each observing run, the coincidence / localization / injection tables are
joined on ``coinc_event_id``, the events are classified as BNS, NSBH or BBH
with a source-frame mass threshold, and the sky maps of the selected classes
are copied out. The injection parameters are also written on their own, one
table per run and class.

Zenodo records:

* 22550048 : FullPop population, folder ``fullpop``
* 22555949 : PixelPop population, folder ``pixelpop``

Usage
-----
::

    python observing_scenarios_skymaps.py --record 22550048 --pop pixelpop --data-dir data
    python observing_scenarios_skymaps.py --zip data/runs.zip --pop pixelpop --runs O5a --classes BNS --ns-max-mass 3.0
    python observing_scenarios_skymaps.py --zip data/runs.zip --list

Outputs, under ``--out-dir``:

* ``observing-scenarios.ecsv``: all selected events, with the path of their
  sky map in the ``skymap`` column;
* ``skymaps/<run>/<class>/<coinc_event_id>.fits``;
* ``injections/<run>/injections_<run>_<class>.ecsv``.
"""

import argparse
import hashlib
import logging
import pathlib
import sqlite3
import sys
import zipfile
from functools import reduce
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import numpy as np
import requests
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy.table import QTable, join, vstack
from tqdm.auto import tqdm

ZENODO_API = "https://zenodo.org/api/records"
CHUNK_SIZE = 1 << 20

#: Summary tables joined on ``coinc_event_id`` to build the event table.
SUMMARY_TABLES = ("coincs.dat", "allsky.dat", "injections.dat")
SOURCE_CLASSES = ("BNS", "NSBH", "BBH")

log = logging.getLogger(__name__)


def parse_arguments(argv=None):
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], allow_abbrev=False
    )
    parser.add_argument(
        "--zip",
        type=pathlib.Path,
        help="archive already on disk; the download is skipped",
    )
    parser.add_argument(
        "--record",
        default="22550048",
        help="pinned Zenodo record id (not the concept DOI)",
    )
    parser.add_argument(
        "--file-name",
        default="runs.zip",
        help="name of the file to download from the record",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("data"),
        help="directory the archive is downloaded to",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/output"),
        help="directory the tables and sky maps are written to",
    )
    parser.add_argument(
        "--subdir",
        default="runs",
        help="root folder inside the archive",
    )
    parser.add_argument(
        "--pop",
        choices=["fullpop", "pixelpop"],
        default="fullpop",
        help="population-model folder inside each run",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["O5a", "O5b", "O5c"],
        help="observing runs to unpack",
    )
    parser.add_argument(
        "--detectors",
        default="",
        help="detector-network suffix of the run folders, if any",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=SOURCE_CLASSES,
        default=["BNS", "NSBH"],
        help="source classes to keep",
    )
    parser.add_argument(
        "--mass-threshold",
        type=float,
        default=3.0,
        help="maximum source-frame neutron-star mass, in solar masses",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the runs and population folders in the archive, then exit",
    )
    return parser.parse_args(argv)


def md5sum(path):
    """Return the hex MD5 digest of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file_metadata(record, file_name):
    """Return the Zenodo metadata of one file of a record.

    Raises
    ------
    KeyError
        If the record holds no file of that name.
    """
    response = requests.get(f"{ZENODO_API}/{record}", timeout=60)
    response.raise_for_status()
    files = response.json().get("files", [])
    for entry in files:
        if entry["key"] == file_name:
            return entry
    raise KeyError(
        f"Zenodo record {record} has no file {file_name!r}; "
        f"it holds {sorted(entry['key'] for entry in files)}"
    )


def download(url, path, resume_from):
    """Download ``url`` to ``path``, continuing from byte ``resume_from``."""
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    # A resume is only valid if the server answered 206 Partial Content.
    # On 200 it is sending the whole file, so appending would corrupt it.
    if resume_from and response.status_code != 206:
        log.warning("server ignored the range request; restarting the download")
        resume_from = 0

    total = int(response.headers.get("content-length", 0)) + resume_from
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(path, "ab" if resume_from else "wb") as stream,
        tqdm(
            total=total or None,
            initial=resume_from,
            unit="B",
            unit_scale=True,
            desc=path.name,
        ) as progress,
    ):
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            stream.write(chunk)
            progress.update(len(chunk))


def fetch(record, file_name, data_dir):
    """Download one file of a Zenodo record and verify its checksum.

    A file already on disk with the right digest is kept as is, and a shorter
    one is resumed.

    Returns
    -------
    pathlib.Path
        Path of the verified file.

    Raises
    ------
    ValueError
        If the file does not match the published MD5 digest.
    """
    path = pathlib.Path(data_dir) / file_name
    metadata = fetch_file_metadata(record, file_name)
    algorithm, _, digest = metadata.get("checksum", "").partition(":")
    if algorithm != "md5" or not digest:
        raise ValueError(f"record does not publish an MD5 checksum: {algorithm!r}")
    size = metadata.get("size")

    resume_from = 0
    if path.exists():
        local = path.stat().st_size
        # Only hash a file that is already the full length; on an archive of
        # tens of GB, hashing a partial one would cost minutes for nothing.
        if size is not None and local < size:
            log.info("%s is short (%d of %d bytes), resuming", path, local, size)
            resume_from = local
        elif md5sum(path) == digest:
            log.info("%s is already present and its MD5 matches", path)
            return path
        else:
            log.warning("%s does not match its MD5, downloading it again", path)

    download(metadata["links"]["self"], path, resume_from)
    found = md5sum(path)
    if found != digest:
        raise ValueError(f"{path}: MD5 is {found}, expected {digest}")
    log.info("downloaded %s, MD5 %s", path, found)
    return path


def list_archive(zip_path, subdir="runs"):
    """Print the run and population folders found in the archive."""
    with zipfile.ZipFile(zip_path) as archive:
        root = zipfile.Path(archive) / subdir
        for run in sorted(p.name for p in root.iterdir() if p.is_dir()):
            pops = sorted(p.name for p in (root / run).iterdir() if p.is_dir())
            print(f"{run}: {', '.join(pops)}")


def read_effective_rate(in_run):
    """Return the rate density the injections of a run were drawn at.

    Parameters
    ----------
    in_run : zipfile.Path
        Population-model folder of the run inside the archive.

    Returns
    -------
    astropy.units.Quantity
    """
    # sqlite3 needs a real file, so the database is spilled out of the zip.
    with (
        (in_run / "events.sqlite").open("rb") as in_file,
        NamedTemporaryFile() as out_file,
    ):
        copyfileobj(in_file, out_file)
        out_file.flush()
        with sqlite3.connect(f"file:{out_file.name}?mode=ro", uri=True) as db:
            ((comment,),) = db.execute(
                "SELECT comment FROM process WHERE program = 'bayestar-inject'"
            )
    return u.Quantity(comment)


def read_run(in_run, run):
    """Read and join the summary tables of one observing run.

    Parameters
    ----------
    in_run : zipfile.Path
        Population-model folder of the run inside the archive.
    run : str
        Run label, written into the ``run`` column.

    Returns
    -------
    table : astropy.table.QTable
        The joined table, carrying the effective rate in its metadata.
    injection_columns : list of str
        Columns that come from ``injections.dat``.
    """
    tables = [
        QTable.read((in_run / filename).read_text(), format="ascii")
        for filename in SUMMARY_TABLES
    ]
    table = reduce(join, tables)
    table["run"] = run
    table.meta.clear()
    table.meta["effective_rate"] = {run: read_effective_rate(in_run)}
    return table, tables[-1].colnames


def classify(table, mass_threshold):
    """Add the ``redshift`` and ``source_class`` columns.

    The threshold is a source-frame one, so it is redshifted before being
    compared to the catalogue's detector-frame masses.
    """
    table["redshift"] = z_at_value(
        cosmo.luminosity_distance, table["distance"] * u.Mpc
    ).to_value(u.dimensionless_unscaled)
    limit = mass_threshold * (1 + table["redshift"])
    table["source_class"] = np.where(
        table["mass1"] <= limit,
        "BNS",
        np.where(table["mass2"] <= limit, "NSBH", "BBH"),
    )
    return table


def process_zip(
    zip_path,
    runs,
    out_dir,
    classes=("BNS", "NSBH"),
    mass_threshold=3.0,
    subdir="runs",
    detectors="",
    pop="fullpop",
):
    """Unpack the archive into event tables, injection tables and sky maps.

    Parameters
    ----------
    zip_path : str or pathlib.Path
        Archive to read.
    runs : sequence of str
        Observing runs to unpack.
    out_dir : str or pathlib.Path
        Destination directory.
    classes : sequence of str, optional
        Source classes to keep, among BNS, NSBH and BBH.
    mass_threshold : float, optional
        Maximum source-frame neutron-star mass, in solar masses.
    subdir : str, optional
        Root folder inside the archive.
    detectors : str, optional
        Detector-network suffix of the run folders, if any.
    pop : str, optional
        Population-model folder inside each run.

    Returns
    -------
    pathlib.Path
        Path of the event table that was written.
    """
    out_root = pathlib.Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        in_root = zipfile.Path(archive) / subdir

        tables, injection_columns = [], []
        for run in tqdm(runs, desc="Reading summary tables"):
            in_run = in_root / f"{run}{detectors}" / pop
            if not in_run.exists():
                raise FileNotFoundError(
                    f"{in_run.at} is not in the archive; use --list to see "
                    "the runs and population folders it holds"
                )
            run_table, columns = read_run(in_run, run)
            tables.append(run_table)
            injection_columns += [c for c in columns if c not in injection_columns]

        # vstack keeps only one run's metadata, so the rates are merged here.
        table = vstack(tables, metadata_conflicts="silent")
        table.meta["effective_rate"] = {
            run: rate for t in tables for run, rate in t.meta["effective_rate"].items()
        }
        # The merger-rate prior depends on the population model, so the model
        # travels with the events rather than being restated downstream.
        table.meta["population_model"] = pop

        table = classify(table, mass_threshold)
        table = table[np.isin(table["source_class"], classes)]

        skymaps = []
        for row in tqdm(table, desc="Copying sky maps"):
            filename = f"{row['coinc_event_id']}.fits"
            in_path = in_root / f"{row['run']}{detectors}" / pop / "allsky" / filename
            out_path = out_root / "skymaps" / row["run"] / row["source_class"]
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path / filename
            with in_path.open("rb") as in_file, out_path.open("wb") as out_file:
                copyfileobj(in_file, out_file)
            skymaps.append(str(out_path.relative_to(out_root)))
        table["skymap"] = skymaps

    event_table = out_root / "observing-scenarios.ecsv"
    table.write(event_table, overwrite=True)
    log.info("wrote %s (%d events)", event_table, len(table))

    columns = injection_columns + ["redshift", "source_class", "skymap"]
    for run in runs:
        for source_class in classes:
            selected = table[
                (table["run"] == run) & (table["source_class"] == source_class)
            ]
            log.info("%s %s: %d events", run, source_class, len(selected))
            if not len(selected):
                continue
            injections = selected[columns]
            injections.meta = {
                "effective_rate": table.meta["effective_rate"][run],
                "population_model": pop,
            }
            out_path = out_root / "injections" / run
            out_path.mkdir(parents=True, exist_ok=True)
            injections.write(
                out_path / f"injections_{run}_{source_class}.ecsv", overwrite=True
            )

    return event_table


def main(argv=None):
    """Download the archive if needed, then unpack it."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_arguments(argv)
    try:
        zip_path = args.zip or fetch(args.record, args.file_name, args.data_dir)
        if args.list:
            list_archive(zip_path, args.subdir)
            return 0
        process_zip(
            zip_path=zip_path,
            runs=args.runs,
            out_dir=args.out_dir,
            classes=args.classes,
            mass_threshold=args.mass_threshold,
            subdir=args.subdir,
            detectors=args.detectors,
            pop=args.pop,
        )
    except Exception as error:  # noqa: BLE001 -- any failure must exit non-zero
        log.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
