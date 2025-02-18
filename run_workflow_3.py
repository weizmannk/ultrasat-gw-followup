"""
ULTRASAT Workflow Execution Script

This script automates the third stage of the ULTRASAT observing scenario simulation workflow, focusing on computing follow-up coverage for gravitational wave (GW) events.

Key Features:
1. Reads observation parameters from a `.ini` configuration file.
2. Executes `./workflow/compute_tiling.py` to determine ULTRASAT follow-up coverage.
3. Runs `./workflow/make-coverage-plots.py` to generate statistics and visualizations.

Assumptions:
- The parameter file includes required observation constraints and mission settings.
- GW sky maps and event schedules are available for processing.

Usage:
    python3 run_workflow_3.py --params /path/to/params_file.ini [--log_dir /path/to/logs]

Example:
    python3 run_workflow_3.py --params ./params-O5.ini
"""


import os
import sys
import subprocess
import configparser
import argparse
import logging
from pathlib import Path
from m4opt.utils.console import status


def setup_logging(log_dir):
    """
    Configure logging for the script.

    Parameters:
        log_dir (str): Directory where log files will be stored.
    """
    log_file = os.path.join(log_dir, "workflow3.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def run_compute_tiling(params_file, followup_dir):
    """
    Run the ./workflow/compute_tiling.py script to compute ULTRASAT follow-up coverage.

    Parameters:
        params_file (str): Path to the params file (absolute path).
        followup_dir (str): Absolute path to the 'ultrasat-gw-followup' directory.
    """

    with status("Compute the Misssion follow-up coverage"):
        compute_tiling_script = os.path.join(followup_dir, "workflow/compute_tiling.py")

        if not os.path.exists(compute_tiling_script):
            logging.error(
                f"./workflow/compute_tiling.py script not found: {compute_tiling_script}"
            )
            sys.exit(1)

        try:
            result = subprocess.run(
                ["python3", compute_tiling_script, params_file],
                check=True,
                text=True,
                # capture_output=True
            )
            logging.info("./workflow/compute_tiling.py completed successfully.")
            logging.debug(f"./workflow/compute_tiling.py output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error("Error running ./workflow/compute_tiling.py:")
            logging.error(e.stderr)
            sys.exit(1)


def run_make_coverage_plots(params_file, followup_dir):
    """
    Run the make-coverage-plots.py script to generate statistics and plots.

    Parameters:
        params_file (str): Path to the params file (absolute path).
        followup_dir (str): Absolute path to the 'ultrasat-gw-followup' directory.
    """
    with status("Generate Statistics and Plots"):
        make_plots_script = os.path.join(
            followup_dir, "workflow/make-coverage-plots.py"
        )

        if not os.path.exists(make_plots_script):
            logging.error(
                f"./workflow/make-coverage-plots.py script not found: {make_plots_script}"
            )
            sys.exit(1)

        try:
            result = subprocess.run(
                ["python3", make_plots_script, params_file],
                check=True,
                text=True,
                # capture_output=True
            )
            logging.info("./workflow/make-coverage-plots.py completed successfully.")
            logging.debug(f"./workflow/make-coverage-plots.py output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error("Error running ./workflow/make-coverage-plots.py:")
            logging.error(e.stderr)
            sys.exit(1)


def create_directories(outdir, additional_dirs=None):
    """
    Create necessary directories for the workflow.

    Parameters:
        outdir (str): Base output directory.
        additional_dirs (list, optional): Additional directories to create within outdir.
    """
    directories = [
        outdir,
        os.path.join(outdir, "coverage_plots"),
        os.path.join(outdir, "statistics"),
    ]
    if additional_dirs:
        for additional_dir in additional_dirs:
            directories.append(os.path.join(outdir, additional_dir))

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Ensured directory exists: {directory}")
        except Exception as e:
            logging.error(f"Failed to create directory {directory}: {e}")
            sys.exit(1)


def read_params_file(params_file):
    """
    Read and parse the parameters from the .ini file.

    Parameters:
        params_file (str): Path to the params file.

    Returns:
        dict: Dictionary containing parameter values.
    """
    with status("Reading parameters from .ini file"):
        config = configparser.ConfigParser()
        config.read(params_file)

        try:
            params = {
                "obs_scenario_dir": config.get("params", "obs_scenario"),
                "save_directory": config.get("params", "save_directory"),
                "bandpass": config.get("params", "bandpass"),
            }
            logging.debug(f"Parameters read from {params_file}: {params}")
            return params
        except (
            configparser.NoSectionError,
            configparser.NoOptionError,
            ValueError,
        ) as e:
            logging.error(f"Error reading parameters from {params_file}: {e}")
            sys.exit(1)


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the third part of the ULTRASAT workflow to compute follow-up coverage and generate plots."
    )
    parser.add_argument(
        "-p", "--params", type=str, required=True, help="Path to the params file."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs3",
        help="Directory for log files (default: ./Logs3).",
    )
    return parser.parse_args()


def main():
    """
    Main function to execute the workflow.
    """
    logging.info("Starting the third part of the ULTRASAT workflow.")

    args = parse_arguments()

    # Read parameters from the .ini file first to extract necessary directories
    params = read_params_file(params_file)

    # Convert directories to absolute paths
    outdir = os.path.abspath(params["save_directory"])
    obs_scenario_dir = os.path.abspath(params["obs_scenario_dir"])

    # Ensure log directory exists before setting up logging
    log_dir = os.path.join(outdir, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir)

    # Get the path to the 'ultrasat-gw-followup' directory
    followup_dir = os.path.dirname(params_file)

    # Run compute_tiling.py
    run_compute_tiling(params_file, followup_dir)

    # Run make-coverage-plots.py
    run_make_coverage_plots(params_file, followup_dir)

    logging.info("Statistics and Plots have been Done!")


if __name__ == "__main__":
    main()
