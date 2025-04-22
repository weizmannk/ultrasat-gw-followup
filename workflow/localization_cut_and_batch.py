#!/usr/bin/env python3

"""
ULTRASAT Follow-Up Data Preprocessing

This script preprocesses ULTRASAT follow-up data for an LVK observing run.
It filters simulated events based on sky localization area (`max_area`) and
batches them for submission, grouping events into `N_batch` files.

Usage:
    python3 localization_cut_and_batch.py /path/to/params.ini
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
from m4opt.utils.console import status
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


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

    bns_mask = (source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)
    nsbh_mask = (source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)
    bbh_mask = (source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)

    return bns_mask, nsbh_mask, bbh_mask


def downselect_and_batch(
    allsky_file,
    injections_file,
    outdir,
    max_area=200,
    N_batch=20,
    BNS=True,
    NSBH=True,
    BBH=False,
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
    :params BNS, NSBH, BBH: Whether to split the population into BNS or/and  NSBH only (excluding BBH)
    :type BNS, NSBH, BBH: bool
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

        allsky = Table.read(allsky_file, format="ascii.fast_tab")
        injections = Table.read(injections_file, format="ascii.fast_tab")

        # Filtering selected populations
        populations = [
            pop_name
            for pop_name, pop_value in zip(["BNS", "NSBH", "BBH"], [BNS, NSBH, BBH])
            if pop_value
        ]

        if not populations:
            logging.error(
                "All populations BNS, NSBH, and BBH are set to False. "
                "Set at least one population to True to run the script."
            )
            sys.exit(1)

        # Classifying populations and  Selecting populations

        populations = [
            pop_name
            for pop_name, pop_value in zip(["BNS", "NSBH", "BBH"], [BNS, NSBH, BBH])
            if pop_value
        ]
        allsky_filename = f"allsky_{'_'.join(populations).lower()}"

        logging.info(f"Using populations: {', '.join(populations)}.")

        bns_mask, nsbh_mask, bbh_mask = classify_populations(injections)

        selected_masks = []
        if "BNS" in populations:
            selected_masks.append(bns_mask)
        if "NSBH" in populations:
            selected_masks.append(nsbh_mask)
        if "BBH" in populations:
            selected_masks.append(bbh_mask)

        # Combining selected masks
        if selected_masks:
            combined_mask = np.logical_or.reduce(selected_masks)
            events = allsky[combined_mask].to_pandas()

            # print(f"event lenght "len{events}")
            events.to_csv(os.path.join(outdir, f"{allsky_filename}.csv"), index=False)
        else:
            logging.warning("No population selected after filtering.")

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
        if events_filtered.empty:
            logging.info("No events found with area <= max_area.")
            return

        percent_filtered = len(events_filtered) / len(events) * 100

        # Log summary of the filtered events
        logging.info(
            f"{percent_filtered:.2f}% of events have localization < {max_area:.1f} sq. deg."
        )
        logging.info(f"Total filtered events: {len(events_filtered)}")

        # Save the filtered events
        savepath = os.path.join(outdir, "allsky_cut.csv")
        events_filtered.to_csv(savepath, index=False)  # , sep=' ')
        logging.info(f"Filtered events saved to {savepath}.")

        # Create a subdirectory for batch files
        batch_dir = os.path.join(outdir, "batches")
        os.makedirs(batch_dir, exist_ok=True)

        # Split filtered events into batches
        logging.info("Batching process")
        batches = np.array_split(events_filtered, N_batch)
        for i, batch in enumerate(batches):
            batch_filename = f"allsky_batch{i}.csv"
            batch_path = os.path.join(batch_dir, batch_filename)
            batch.to_csv(batch_path, index=False)  # sep=',')
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
        BNS = config.getboolean("params", "BNS", fallback=True)
        NSBH = config.getboolean("params", "NSBH", fallback=True)
        BBH = config.getboolean("params", "BBH", fallback=False)
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
        BNS=BNS,
        NSBH=NSBH,
        BBH=BBH,
    )
