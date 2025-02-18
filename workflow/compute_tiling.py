#!/usr/bin/env python
# coding: utf-8

"""
Script: compute_ultrasat_tiling.py
==================================
Description:
------------
This script computes the number of ULTRASAT pointings required to tile a given
gravitational wave (GW) localization region using sky maps. It processes
observing schedules, sky maps (FITS files), and schedules (ECSV files)
to calculate the probability coverage and number of tiles needed to reach 90% confidence.

It integrates the ULTRASAT field of view (FoV) and calculates the sky coverage
based on the scheduled pointings.

Features:
---------
- Reads all-sky event schedules.
- Loads and rasterizes GW sky maps.
- Reads observing schedules and extracts pointing coordinates.
- Computes the sky coverage for each event.
- Outputs a summary file with coverage statistics.

Usage:
------
    $ python3 compute_tiling.py /path/to/params_file.ini

Outputs:
--------
- A text file `allsky_coverage.csv` in the output directory with:
  - Event ID
  - Percent coverage
  - Scheduled exposure time
  - Number of tiles needed for 90% confidence
  - Total number of tiles for a unique visit
  - Total number of visits per target coordinate

"""


#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import configparser
import logging
import numpy as np
import pandas as pd

from astropy.table import QTable
from astropy.coordinates import ICRS
from astropy_healpix import HEALPix

from ligo.skymap.tool import ArgumentParser
from ligo.skymap.io import read_sky_map
from ligo.skymap.bayestar import rasterize

from m4opt.fov import footprint_healpix
from m4opt.utils.console import status
from m4opt.missions import ultrasat

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def compute_tiling(
    allsky_sched: str, skymap_dir: str, sched_dir: str, outdir: str, nside=128
):
    """
    Computes the number of ULTRASAT pointings needed to tile a gravitational wave localization region.

    Parameters:
    -----------------------------
    allsky_sched (str): Path to allsky scheduled events file.
    fitsloc (str): Path to directory containing sky maps.
    schedloc (str): Path to directory containing schedules.
    outdir (str): Output directory for results.

    Returns:
    -----------------------------
    Saves computed tiling statistics to a file.
    """

    hpx = HEALPix(nside, frame=ICRS(), order="nested")

    # Define sky region using ULTRASAT field of view
    sky_region = ultrasat.fov

    with status("Reading event schedule..."):
        try:
            event_df = pd.read_csv(allsky_sched)
        except Exception as e:
            logging.error(f"Failed to read allsky schedule file {allsky_sched}: {e}")
            sys.exit(1)

    event_list = event_df["event_id"].tolist()
    texp_list = event_df["t_exp (ks)"].tolist()

    skymap_list = [os.path.join(skymap_dir, f"{event}.fits") for event in event_list]
    sched_list = [os.path.join(sched_dir, f"{event}.ecsv") for event in event_list]

    results = []
    with status(f"Processing {len(event_list)} events..."):
        for event_id, skymap_file, schedule_file, texp in zip(
            event_list, skymap_list, sched_list, texp_list
        ):

            with status(f"Reading sky map for event {event_id}..."):
                try:
                    skymap_moc = read_sky_map(skymap_file, moc=True)[
                        "UNIQ", "PROBDENSITY"
                    ]
                    skymap_prob = rasterize(skymap_moc, hpx.level)["PROB"]
                except Exception as e:
                    logging.warning(
                        f"Skipping event {event_id} due to an issue reading FITS file: {e}"
                    )
                    continue

            if not os.path.isfile(schedule_file):
                logging.warning(
                    f"Schedule file not found for event {event_id}: {schedule_file}"
                )
                continue

            if os.stat(schedule_file).st_size == 0:
                logging.warning(
                    f"Skipping event {event_id} due to an empty schedule file."
                )
                continue

            with status(f"Reading schedule for event {event_id}..."):
                try:
                    schedule = QTable.read(schedule_file, format="ascii.ecsv")
                except Exception as e:
                    logging.error(f"Failed to read schedule file {schedule_file}: {e}")
                    continue

            indices = np.array([], dtype=np.intp)
            tiles_to_90pct: int = 0
            row_count: int = 0
            reached_90 = False

            visit_counts = {}
            seen_coords = set()  # Tarck unique coordinates
            with status(f"Computing coverage for event {event_id}..."):
                for row in schedule:
                    if row["action"] == "slew":
                        logging.info(f"Slew action detected, skipping this row.")
                        continue

                    target_coord = row["target_coord"]
                    roll = row["roll"]

                    # Convert SkyCoord to tuple (RA, Dec)
                    coord_tuple = (
                        float(target_coord.ra.deg),
                        float(target_coord.dec.deg),
                    )
                    visit_counts[coord_tuple] = visit_counts.get(coord_tuple, 0) + 1

                    # Count only the first visit of each unique coordinate
                    if coord_tuple not in seen_coords:
                        seen_coords.add(coord_tuple)
                        row_count += 1

                    # Compute new indices for footprint
                    new_indices = footprint_healpix(
                        hpx=hpx,
                        region=sky_region,
                        target_coord=target_coord,
                        rotation=roll,
                    )
                    indices = np.unique(np.concatenate((indices, new_indices)))

                    # Compute cumulative probability
                    prob_cum = 100 * skymap_prob[indices].sum()

                    # Stop counting tiles once we reach 90% probability
                    if prob_cum > 90 and not reached_90:
                        reached_90 = True
                        tiles_to_90pct = row_count

            unique_visit_counts = set(visit_counts.values())

            if len(unique_visit_counts) > 1:
                logging.warning(
                    f"Some coordinates have a different number of visits!\n {visit_counts}"
                )
                visit_total = None  # Set to None if visits are not uniform
            else:
                visit_total = next(iter(unique_visit_counts), 0)

            tiles_total = row_count
            total_prob = 100 * skymap_prob[indices].sum()
            results.append(
                [
                    event_id,
                    total_prob,
                    texp,
                    int(tiles_to_90pct),
                    tiles_total,
                    visit_total,
                ]
            )

            logging.info(f"Percent coverage: {total_prob:.2f}% for event {event_id}")

    with status("Saving results..."):
        results_df = pd.DataFrame(
            results,
            columns=[
                "event_id",
                "percent_coverage",
                "texp_sched (ks)",
                "tiles_to_90pct",
                "tiles_total",
                "visit_total",
            ],
        )

        output_file = os.path.join(outdir, "allsky_coverage.csv")
        try:
            results_df.to_csv(output_file, index=False)
            logging.info(f"Saved results to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")

    return


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Compute tiling statistics for GW sky maps."
    )
    parser.add_argument(
        "params", type=str, help="Path to parameters file (.ini format)."
    )
    args = parser.parse_args()

    # Read configuration
    config = configparser.ConfigParser()
    try:
        config.read(args.params)
    except Exception as e:
        logging.error(f"Failed to read parameters file {args.params}: {e}")
        sys.exit(1)

    try:
        obs_scenario_dir = config.get("params", "obs_scenario")
        outdir = config.get("params", "save_directory")
        nside = config.getint("params", "nside")

    except Exception as e:
        logging.error(f"Missing required parameters in config file: {e}")
        sys.exit(1)

    allsky_sched = os.path.join(outdir, "allsky_sched_full.csv")
    skymap_dir = os.path.join(obs_scenario_dir, "allsky/")
    sched_dir = os.path.join(outdir, "schedules/")

    # Run script
    compute_tiling(allsky_sched, skymap_dir, sched_dir, outdir, nside)
