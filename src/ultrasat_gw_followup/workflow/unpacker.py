"""Unpack the Zenodo archive into an event table and one sky map per event.

The archive holds, per observing run, the coincidence / localization /
injection tables and the all-sky FITS maps. This step joins the three tables,
keeps the BNS and NSBH events (secondary mass below the threshold, in the
source frame) and copies their sky maps out.

Usage
-----
::

    python -m ultrasat_gw_followup.workflow.unpacker --config config/ultrasat_lcs.ini
    ugf-unpacker --zip data/runs.zip --runs O5a O5b O5c --data-dir data

Outputs: ``<data_dir>/observing-scenarios.ecsv`` and
``<data_dir>/<skymap_dir>/<run>/<coinc_event_id>.fits``.
"""

import argparse
import logging
import pathlib
import sqlite3
import sys
import zipfile
from functools import reduce
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy.table import QTable, join, vstack
from tqdm.auto import tqdm

from ultrasat_gw_followup.config import Config

#: Summary tables joined on ``coinc_event_id`` to build the event table.
SUMMARY_TABLES = ("coincs.dat", "allsky.dat", "injections.dat")

log = logging.getLogger(__name__)


def parse_arguments(argv=None):
    """Parse the command line, with the ``.ini`` values as defaults."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], allow_abbrev=False
    )
    config, remaining = Config.from_command_line(parser, argv)

    parser.add_argument(
        "--zip",
        type=pathlib.Path,
        default=config.get("scenarios", "zip", "data/runs.zip"),
        help="Zenodo archive to unpack",
    )
    parser.add_argument(
        "--subdir",
        default=config.get("scenarios", "subdir", "runs"),
        help="root folder inside the archive",
    )
    parser.add_argument(
        "--pop",
        choices=["fullpop", "pixelpop"],
        default=config.get("scenarios", "pop", "fullpop"),
        help="population-model folder inside each run",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=config.getlist("scenarios", "runs", ["O5a", "O5b", "O5c"]),
        help="observing runs to unpack",
    )
    parser.add_argument(
        "--detectors",
        default=config.get("scenarios", "detectors", ""),
        help="detector-network suffix of the run folders, if any",
    )
    parser.add_argument(
        "--mass-threshold",
        type=float,
        default=config.getfloat("scenarios", "mass_threshold", 3.0),
        help="maximum source-frame secondary mass, in solar masses",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path(config.get("paths", "data_dir", "data")),
        help="directory the event table is written to",
    )
    parser.add_argument(
        "--skymap-dir",
        default=config.get("paths", "skymap_dir", "skymaps"),
        help="sky map directory, relative to --data-dir",
    )
    return parser.parse_args(remaining)


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
    astropy.table.QTable
        The joined table, carrying the simulation's effective rate in its
        metadata.
    """
    table = reduce(
        join,
        (
            QTable.read((in_run / filename).read_text(), format="ascii")
            for filename in SUMMARY_TABLES
        ),
    )
    table["run"] = run
    table.meta.clear()
    table.meta["effective_rate"] = {run: read_effective_rate(in_run)}
    return table


def read_effective_rate(in_run):
    """Return the effective rate of a simulation, from its sqlite provenance.

    Parameters
    ----------
    in_run : zipfile.Path
        Population-model folder of the run inside the archive.

    Returns
    -------
    astropy.units.Quantity
        Rate density the injections were drawn at.
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


def process_zip(
    zip_path,
    runs,
    data_dir,
    skymap_dir,
    mass_threshold=3.0,
    subdir="runs",
    detectors="",
    pop="fullpop",
):
    """Unpack the archive into an event table and the matching sky maps.

    Parameters
    ----------
    zip_path : str or pathlib.Path
        Archive to read.
    runs : sequence of str
        Observing runs to unpack.
    data_dir : str or pathlib.Path
        Destination directory.
    skymap_dir : str
        Sky map directory, relative to ``data_dir``.
    mass_threshold : float, optional
        Maximum source-frame secondary mass, in solar masses.
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
    out_root = pathlib.Path(data_dir)
    skymap_root = out_root / skymap_dir
    for run in runs:
        (skymap_root / run).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        in_root = zipfile.Path(archive) / subdir

        table = vstack(
            [
                read_run(in_root / f"{run}{detectors}" / pop, run)
                for run in tqdm(runs, desc="Reading summary tables")
            ]
        )

        # Keep the BNS and NSBH events: the mass threshold is a source-frame
        # one, so it is redshifted before being compared to the catalogue's
        # detector-frame masses.
        table["redshift"] = z_at_value(
            cosmo.luminosity_distance, table["distance"] * u.Mpc
        ).to_value(u.dimensionless_unscaled)
        table = table[table["mass2"] <= mass_threshold * (1 + table["redshift"])]
        table["source_class"] = np.where(
            table["mass1"] <= mass_threshold * (1 + table["redshift"]), "BNS", "NSBH"
        )

        event_table = out_root / "observing-scenarios.ecsv"
        table.write(event_table, overwrite=True)

        for row in tqdm(table, desc="Copying sky maps"):
            filename = f"{row['coinc_event_id']}.fits"
            in_path = in_root / f"{row['run']}{detectors}" / pop / "allsky" / filename
            out_path = skymap_root / row["run"] / filename
            with in_path.open("rb") as in_file, out_path.open("wb") as out_file:
                copyfileobj(in_file, out_file)

    log.info("wrote %s (%d events)", event_table, len(table))
    return event_table


def main(argv=None):
    """Entry point of ``ugf-unpacker``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_arguments(argv)
    process_zip(
        zip_path=args.zip,
        runs=args.runs,
        data_dir=args.data_dir,
        skymap_dir=args.skymap_dir,
        mass_threshold=args.mass_threshold,
        subdir=args.subdir,
        detectors=args.detectors,
        pop=args.pop,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
