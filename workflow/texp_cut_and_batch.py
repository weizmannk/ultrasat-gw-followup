#!/usr/bin/env python3

"""
This Python script processes exposure time data from ULTRASAT follow-up simulations
for an LVK observing run scenario. It performs the following steps:

1. Loads and aggregates exposure time results from multiple batch files.
2. Ensures that all exposure times meet a minimum threshold (`min_texp`).
3. Converts exposure times from seconds to kiloseconds (ks).
4. Saves the processed data into a master output file (`allsky_sched_full.csv`).
5. Splits the dataset into smaller batch files for scheduling.

The script reads configuration parameters from an 'ini' file and automatically
creates necessary output directories.

Usage           : python3 texp_cut_and_batch.py /path/to/params.ini

Parameters in INI File:
    - save_directory  : Output directory where processed files will be stored.
    - N_batch_preproc : Number of input batch files from `max-texp-by-sky-loc.py` (default: 1).
    - N_batch_sched   : Number of output batch files for scheduling (default: `N_batch_preproc`).
    - band            : UV band being considered (`nuv`).
    - min_texp        : Minimum allowed exposure time in seconds.

Output Files:
    - allsky_sched_full.csv    : Processed exposure times for all events.
    - texp_sched/allsky_sched_batch*.csv : Batched exposure time files for scheduling.

"""

import logging
import os
import pandas as pd
import numpy as np
import sys
import configparser
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def texp_cut_and_batch(texp_dir, out_dir, N_in, N_out, bandpass, min_texp):
    """
    Processes exposure time data from texp_cut_and_batch.py output and prepares batch files.

    Steps:
        1. Load and aggregate event files from multiple batches.
        2. Ensure all exposure times are at least `min_texp` seconds.
        3. Convert exposure times from seconds to kiloseconds (ks).
        4. Save the processed data and split it into batches for the scheduler.

    Parameters:
    -----------------------
    texp_dir (str)   : Path to the directory containing input event files.
    out_dir (str)    : Path to save output files.
    N_in (int)       : Number of input batches (from texp_cut_and_batch.py).
    N_out (int)      : Number of output batches (for the scheduler).
    bandpass (str)   : UV band ('nuv') used for parsing filenames.
    min_texp (float) : Minimum exposure time in seconds. Any lower values will be set to this.

    Returns:
    -----------------------
    None
    """

    all_batches = []

    # Load and concatenate all batch files
    for i in range(N_in):
        batch_file = os.path.join(
            texp_dir, f"allsky_texp_max_cut_{bandpass}_batch{i}.csv"
        )

        if not os.path.exists(batch_file):
            logging.warning(f"File {batch_file} not found. Skipping.")
            continue

        batch_data = pd.read_csv(batch_file, delimiter=",")
        print(batch_data)
        all_batches.append(batch_data)

    if not all_batches:
        logging.error("No valid batch files found. Exiting.")
        sys.exit(1)

    # Remove empty DataFrames before concatenation
    all_batches = [df for df in all_batches if not df.empty]

    # Merge all batch data
    events = pd.concat(all_batches, ignore_index=True)

    # Ensure minimum exposure time and convert to kiloseconds (ks)
    events["t_exp (ks)"] = (
        np.where(events["texp_max (s)"] <= min_texp, min_texp, events["texp_max (s)"])
        / 1000
    )

    # Save processed data
    full_output_path = os.path.join(out_dir, "allsky_sched_full.csv")
    events[["event_id", "t_exp (ks)"]].to_csv(
        full_output_path, index=False
    )  # , sep=' ')
    logging.info(f"Processed data saved to: {full_output_path}")

    # Create output directory for batch files
    batch_dir = os.path.join(out_dir, "texp_sched")
    os.makedirs(batch_dir, exist_ok=True)

    # Split dataset into batches
    list_of_batches = np.array_split(events[["event_id", "t_exp (ks)"]], N_out)

    for i, batch in enumerate(list_of_batches):
        batch_file = os.path.join(batch_dir, f"allsky_sched_batch{i}.csv")
        batch.to_csv(batch_file, index=False)  # , sep=',')
        logging.info(f"Batch file created: {batch_file}")

    logging.info("Batching process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process exposure time data and prepare it for ULTRASAT scheduling."
    )
    parser.add_argument(
        "params",
        type=str,
        help="Path to the parameters file (e.g., /path/to/params_file.ini)",
    )
    args = parser.parse_args()

    # Load configuration
    config = configparser.ConfigParser()
    config.read(args.params)

    try:
        out_dir = config.get("params", "save_directory")
        N_in = config.getint("params", "N_batch_preproc", fallback=1)
        N_out = config.getint("params", "N_batch_sched", fallback=N_in)
        bandpass = config.get("params", "bandpass")
        min_texp = config.getfloat("params", "min_texp")
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        logging.error(f"Error reading configuration: {e}")
        sys.exit(1)

    # Define directory for input files
    texp_dir = os.path.join(out_dir, "texp_out")
    os.makedirs(texp_dir, exist_ok=True)

    # Run processing function
    texp_cut_and_batch(texp_dir, out_dir, N_in, N_out, bandpass, min_texp)
