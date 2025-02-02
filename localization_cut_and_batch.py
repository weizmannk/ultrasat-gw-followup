#!/usr/bin/env python3

"""
---------------------------------------------------------------------------------------------------
ABOUT THE SCRIPT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository URL  : https://github.com/weizmannk/ultrasat-gw-followup.git
Creation Date   : January 2024
Description     : This Python script preprocesses ULTRASAT follow-up data for an LVK observing run
                  scenario. It filters simulated events based on sky localization area less than
                  a specified maximum (`max_area`), and batches them for submission, creating
                  batch files with a specified number of events per batch (`N_batch`).
Usage           : python localization_cut_and_batch.py /path/to/params.ini
"""

import os
import sys
import numpy as np
import pandas as pd
import configparser
import argparse
import logging
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo, z_at_value
import astropy.units as u

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def classify_populations(table, ns_max_mass=3.0):
    """
    Splits Compact Binary Coalescence (CBC) events based on source frame mass.

    :param table: Table containing the CBC data
    :type table: astropy.table.Table
    :param ns_max_mass: Maximum neutron star mass threshold
    :type ns_max_mass: float
    :return: Tuple containing boolean masks for BNS, NSBH, and BBH
    :rtype: tuple
    """
    z = z_at_value(cosmo.luminosity_distance, table["distance"] * u.Mpc).to_value(
        u.dimensionless_unscaled
    )
    zp1 = z + 1

    source_mass1 = table["mass1"] / zp1
    source_mass2 = table["mass2"] / zp1

    BNS = (source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)
    NSBH = (source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)
    BBH = (source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)

    return BNS, NSBH, BBH


def downselect_and_batch(
    allsky_file, injections_file, outdir, max_area=200, N_batch=20, split_pop=False
):
    """
    Preprocesses ULTRASAT follow-up data for an LVK observing run scenario by filtering
    events based on sky localization area and batching them for submission.

    :param allsky_file: Path to the all-sky file
    :type allsky_file: str
    :param injections_file: Path to the injections file
    :type injections_file: str
    :param outdir: Path to the output directory
    :type outdir: str
    :param max_area: Maximum sky localization area to trigger on in sq. deg.
    :type max_area: float
    :param N_batch: Number of batch files desired
    :type N_batch: int
    :param split_pop: Whether to split the population into BNS and NSBH only (excluding BBH)
    :type split_pop: bool
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(outdir, exist_ok=True)

        # Verify files exist
        if not os.path.exists(allsky_file):
            logging.error(f"All-sky file '{allsky_file}' does not exist.")
            sys.exit(1)
        if not os.path.exists(injections_file):
            logging.error(f"Injections file '{injections_file}' does not exist.")
            sys.exit(1)

        # Load events data
        if split_pop:
            logging.info(f"Using only BNS and NSBH populations. Excluded BBH.")
            allsky = Table.read(allsky_file, format="ascii.fast_tab")
            injections = Table.read(injections_file, format="ascii.fast_tab")
            BNS, NSBH, BBH = classify_populations(injections)
            events = allsky[BNS | NSBH].to_pandas()

        else:
            logging.info(f"Including all CBC populations (BNS, NSBH, and BBH).")
            events = pd.read_csv(allsky_file, delimiter="\t", skiprows=1)

        # Check required columns
        required_columns = ["area(90)"]
        missing_columns = [col for col in required_columns if col not in events.columns]
        if missing_columns:
            logging.error(
                f"Input file '{allsky_file}' is missing required columns: {missing_columns}"
            )
            sys.exit(1)

        # Filter events by max_area
        logging.info(f"Cutoff event area: {max_area} sq. deg.")
        events_filtered = events[events["area(90)"] <= max_area]
        percent_filtered = len(events_filtered) / len(events) * 100

        # Log summary of the filtered events
        logging.info(
            f"{percent_filtered:.2f}% of events have localization < {max_area:.1f} sq. deg."
        )
        logging.info(f"Total filtered events: {len(events_filtered)}")

        # Save the filtered events
        savepath = os.path.join(outdir, "allsky_cut.txt")
        events_filtered.to_csv(savepath, index=False, sep=" ")
        logging.info(f"Filtered events saved to {savepath}.")

        # Create a subdirectory for batch files
        batch_dir = os.path.join(outdir, "batches")
        os.makedirs(batch_dir, exist_ok=True)

        # Split filtered events into batches
        batches = np.array_split(events_filtered, N_batch)
        for i, batch in enumerate(batches):
            batch_filename = f"allsky_batch{i}.txt"
            batch_path = os.path.join(batch_dir, batch_filename)
            batch.to_csv(batch_path, index=False, sep=",")
            logging.info(f"Batch file created: {batch_path} with {len(batch)} events.")

        logging.info("Batching process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare an LVK observing run scenario for UVEX localization calculations."
    )
    parser.add_argument(
        "params", type=str, help="Path to the params file (e.g., /path/to/params.ini)"
    )

    args = parser.parse_args()

    # Load configuration
    config = configparser.ConfigParser()
    if not os.path.exists(args.params):
        logging.error(f"Params file '{args.params}' does not exist.")
        sys.exit(1)
    config.read(args.params)

    # Extract parameters from the config file
    try:
        obs_scenario_dir = config.get("params", "obs_scenario")
        outdir = config.get("params", "save_directory")
        max_area = config.getfloat("params", "max_area", fallback=2000)
        N_batch = config.getint("params", "N_batch_preproc", fallback=20)
        split_pop = config.getboolean("params", "split_pop", fallback=False)
    except configparser.Error as config_error:
        logging.error(f"Error reading configuration: {config_error}")
        sys.exit(1)

    allsky_file = os.path.join(obs_scenario_dir, "allsky.dat")
    injections_file = os.path.join(obs_scenario_dir, "injections.dat")

    downselect_and_batch(
        allsky_file,
        injections_file,
        outdir,
        max_area=max_area,
        N_batch=N_batch,
        split_pop=split_pop,
    )
