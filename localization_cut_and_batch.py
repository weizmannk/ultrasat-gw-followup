# #!/usr/bin/env python3
# """
# ---------------------------------------------------------------------------------------------------
# ABOUT THE SCRIPT
# ---------------------------------------------------------------------------------------------------
# Author          : Ramodgwendé Weizmann KIENDREBEOGO
# Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
# Repository URL  : https://github.com/weizmannk/ultrasat-gw-followup.git
# Creation Date   : January 2024
# Description     : This Python script preprocesses ULTRASAT follow-up data for an LVK observing run
#                   scenario. It filters simulated events based on sky localization area less than
#                   a specified maximum (`max_area`), and batches them for submission, creating
#                   batch files with a specified number of events per batch (`N_batch`).
# Usage           : python localization_cut_and_batch.py /path/to/params.ini

# This script pulls all simulated events and selects those with a sky localization area
# less than a specified maximum (`max_area`). It then creates files for batch submission
# with a specified number of events per batch (`N_batch`).
# """

# import argparse
# import configparser
# import logging
# import os
# import sys

# import numpy as np
# import pandas as pd

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )


# def downselect_and_batch(
#     allsky_file: str,
#     outdir: str,
#     max_area: float = 200.0,
#     N_batch: int = 20
# ) -> None:
#     """
#     Preprocesses UVEX follow-up data for an LVK observing run scenario by filtering
#     events based on sky localization area and batching them for submission.

#     Args:
#         allsky_file (str): Path to the all-sky file (e.g., '/path/to/allsky.dat').
#         outdir (str): Path to the output directory (e.g., '/path/to/output/directory/').
#         max_area (float, optional): Maximum sky localization area to trigger on in sq. deg.
#                                     Defaults to 200.0.
#         N_batch (int, optional): Number of batches to split the filtered events into.
#                                  Defaults to 20.
#     """
#     try:
#         # Create output directory if it doesn't exist
#         os.makedirs(outdir, exist_ok=True)
#         logging.info(f"Output directory is set to: {outdir}")

#         # Verify allsky_file exists
#         if not os.path.isfile(allsky_file):
#             logging.error(f"All-sky file {allsky_file} does not exist.")
#             sys.exit(1)

#         # Load events data
#         events = pd.read_csv(allsky_file, delimiter='\t', skiprows=1)
#         logging.info(f"Loaded {len(events)} events from {allsky_file}.")

#         # Check for required columns
#         required_columns = ['area(90)']
#         missing_columns = [col for col in required_columns if col not in events.columns]
#         if missing_columns:
#             logging.error(f"Input file {allsky_file} is missing required columns: {missing_columns}")
#             sys.exit(1)

#         # Filter events by max_area
#         events_filtered = events[events['area(90)'] <= max_area]
#         percent_filtered = len(events_filtered) / len(events) * 100

#         # Log summary of the filtered events
#         logging.info(f"{percent_filtered:.2f}% of all events have less than {max_area:.1f} sq. deg. localization.")
#         logging.info(f"Total events with less than {max_area:.1f} sq. deg. localization: {len(events_filtered)}")

#         # Save the filtered events
#         savepath = os.path.join(outdir, 'allsky_cut.txt')
#         events_filtered.to_csv(savepath, index=False, sep=' ')
#         logging.info(f"Filtered events saved to {savepath}.")

#         # Create a subdirectory for batch files
#         batch_dir = os.path.join(outdir, 'batches')
#         os.makedirs(batch_dir, exist_ok=True)
#         logging.info(f"Batch directory is set to: {batch_dir}")

#         # Split filtered events into batches
#         batches = np.array_split(events_filtered, N_batch)
#         for i, batch in enumerate(batches):
#             batch_filename = f'allsky_batch{i}.txt'
#             batch_path = os.path.join(batch_dir, batch_filename)
#             batch.to_csv(batch_path, index=False, sep=',')
#             logging.info(f"Batch file created: {batch_path} with {len(batch)} events.")

#         logging.info("Batching process completed successfully.")

#     except Exception as e:
#         logging.error(f"An error occurred during processing: {e}")
#         sys.exit(1)


# def parse_arguments() -> argparse.Namespace:
#     """
#     Parses command-line arguments.

#     Returns:
#         argparse.Namespace: Parsed arguments.
#     """
#     parser = argparse.ArgumentParser(
#         description="Prepare an LVK observing run scenario for UVEX localization calculations."
#     )
#     parser.add_argument(
#         'params',
#         type=str,
#         help='Path to the params file (e.g., /path/to/params_file.ini)'
#     )
#     return parser.parse_args()


# def load_config(params_path: str) -> configparser.ConfigParser:
#     """
#     Loads the configuration from the given params file.

#     Args:
#         params_path (str): Path to the params file.

#     Returns:
#         configparser.ConfigParser: Parsed configuration.
#     """
#     config = configparser.ConfigParser()
#     if not os.path.isfile(params_path):
#         logging.error(f"Params file '{params_path}' does not exist.")
#         sys.exit(1)
#     config.read(params_path)
#     return config


# def main() -> None:
#     """
#     Main function to execute the downselection and batching process.
#     """
#     args = parse_arguments()

#     # Load configuration
#     config = load_config(args.params)

#     # Extract parameters from the config file
#     try:
#         obs_scenario_dir = config.get("params", "obs_scenario")
#         outdir = config.get("params", "save_directory")
#         max_area = config.getfloat("params", "max_area", fallback=200.0)
#         N_batch = config.getint("params", "N_batch_preproc", fallback=20)
#         logging.info("Configuration parameters loaded successfully.")
#     except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as config_error:
#         logging.error(f"Error reading configuration: {config_error}")
#         sys.exit(1)

#     # Construct the path to the all-sky data file
#     allsky_file = os.path.join(obs_scenario_dir, 'allsky.dat')
#     logging.info(f"All-sky data file path: '{allsky_file}'.")

#     # Run the main processing function
#     downselect_and_batch(allsky_file, outdir, max_area=max_area, N_batch=N_batch)


# if __name__ == '__main__':
#     main()


import numpy as np
import pandas as pd
import os
import sys
import configparser
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def downselect_and_batch(allsky_file, outdir, max_area=100, N_batch=20):
    """
    Function to do preprocessing for UVEX follow-up of an LVK observing run scenario.

    Arguments
    ------------------------
    allsky_file (str) : Path to the all-sky file (e.g., '/path/to/allsky_file.dat')
    outdir (str) : Path to the output directory (e.g., '/path/to/output/directory/')
    max_area (float) : Maximum sky localization area to trigger on in sq. deg. (default is 100)
    N_batch (int) : Number of batch files desired (default is 20)
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(outdir, exist_ok=True)

        # Load events data
        if not os.path.exists(allsky_file):
            logging.error(f"All-sky file {allsky_file} does not exist.")
            sys.exit(1)

        events = pd.read_csv(allsky_file, delimiter="\t", skiprows=1)
        required_columns = ["area(90)"]
        if not all(col in events.columns for col in required_columns):
            logging.error(
                f"Input file {allsky_file} is missing required columns: {required_columns}"
            )
            sys.exit(1)

        # Filter events by max_area
        events_cut = events[events["area(90)"] <= max_area]
        percent_cut = len(events_cut) / len(events) * 100

        # Log summary of the filtered events
        logging.info(
            f"{percent_cut:.2f}% of all events have less than {max_area:.1f} sq. deg. localization."
        )
        logging.info(
            f"Total events with less than {max_area:.1f} sq. deg. localization: {len(events_cut)}"
        )

        # Save the filtered events
        savepath = os.path.join(outdir, "allsky_cut.txt")
        events_cut.to_csv(savepath, index=False, sep=" ")
        logging.info(f"Filtered events saved to {savepath}")

        # Create a subdirectory for batch files
        batch_dir = os.path.join(outdir, "batches")
        os.makedirs(batch_dir, exist_ok=True)

        # Split filtered events into batches
        list_of_batches = np.array_split(events_cut, N_batch)
        for i, batch in enumerate(list_of_batches):
            batch_filename = f"allsky_batch{i}.txt"
            batch_path = os.path.join(batch_dir, batch_filename)
            batch.to_csv(batch_path, index=False, sep=",")
            logging.info(f"Batch file created: {batch_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare an LVK observing run scenario for UVEX localization calculations."
    )
    parser.add_argument(
        "params",
        type=str,
        help="Path to the params file (e.g., /path/to/params_file.ini)",
    )

    args = parser.parse_args()

    # Parse the configuration file
    config = configparser.ConfigParser()
    config.read(args.params)

    # Extract parameters from the config file
    try:
        obs_scenario_dir = config.get("params", "obs_scenario")
        outdir = config.get("params", "save_directory")
        max_area = float(config.get("params", "max_area", fallback=100))
        N_batch = int(config.get("params", "N_batch_preproc", fallback=20))
    except configparser.Error as config_error:
        logging.error(f"Error reading configuration: {config_error}")
        sys.exit(1)

    # Construct the path to the all-sky data file
    allsky_file = os.path.join(obs_scenario_dir, "allsky.dat")

    # Run the main processing function
    downselect_and_batch(allsky_file, outdir, max_area=max_area, N_batch=N_batch)
