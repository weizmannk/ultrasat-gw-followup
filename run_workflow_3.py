#!/usr/bin/env python3
"""
Run the third part of the ULTRASAT observing scenario simulation workflow and compute ULTRASAT follow-up coverage for each event.

Usage:
    python3 run_workflow_3.py --params /path/to/params_file.ini [--log_dir /path/to/logs]

Example:
    python3 run_workflow_3.py --params /home/weizmann.kiendrebeogo/M4OPT-ULTRSASAT/ultrasat-gw-followup/params.ini
"""

import os
import sys
import subprocess
import configparser
import argparse
import logging
from pathlib import Path


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
    Run the compute_tiling.py script to compute ULTRASAT follow-up coverage.

    Parameters:
        params_file (str): Path to the params file (absolute path).
        followup_dir (str): Absolute path to the 'ultrasat-gw-followup' directory.
    """
    logging.info("Running compute_tiling.py to compute ULTRASAT follow-up coverage...")
    compute_tiling_script = os.path.join(followup_dir, "compute_tiling.py")

    if not os.path.exists(compute_tiling_script):
        logging.error(f"compute_tiling.py script not found: {compute_tiling_script}")
        sys.exit(1)

    try:
        result = subprocess.run(
            ["python3", compute_tiling_script, params_file],
            check=True,
            text=True,
            capture_output=True,
        )
        logging.info("compute_tiling.py completed successfully.")
        logging.debug(f"compute_tiling.py output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error("Error running compute_tiling.py:")
        logging.error(e.stderr)
        sys.exit(1)


def run_make_coverage_plots(params_file, followup_dir):
    """
    Run the make-coverage-plots.py script to generate statistics and plots.

    Parameters:
        params_file (str): Path to the params file (absolute path).
        followup_dir (str): Absolute path to the 'ultrasat-gw-followup' directory.
    """
    logging.info("Running make-coverage-plots.py to generate statistics and plots...")
    make_plots_script = os.path.join(followup_dir, "make-coverage-plots.py")

    if not os.path.exists(make_plots_script):
        logging.error(f"make-coverage-plots.py script not found: {make_plots_script}")
        sys.exit(1)

    try:
        result = subprocess.run(
            ["python3", make_plots_script, params_file],
            check=True,
            text=True,
            capture_output=True,
        )
        logging.info("make-coverage-plots.py completed successfully.")
        logging.debug(f"make-coverage-plots.py output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error("Error running make-coverage-plots.py:")
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
    config = configparser.ConfigParser()
    config.read(params_file)

    try:
        params = {
            "obs_scenario_dir": config.get("params", "obs_scenario"),
            "save_directory": config.get("params", "save_directory"),
            "band": config.get("params", "band"),
            # 'MAG': config.getfloat("params", "MAG"),
            # 'RATE_ADJ': config.getfloat("params", "RATE_ADJ"),
            "tiling_time": config.get("params", "tiling_time"),
        }
        logging.debug(f"Parameters read from {params_file}: {params}")
        return params
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
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
    args = parse_arguments()

    # Convert directories to absolute paths
    log_dir = os.path.abspath(args.log_dir)
    params_file = os.path.abspath(args.params)

    # Ensure log directory exists before setting up logging
    try:
        os.makedirs(log_dir, exist_ok=True)
        setup_logging(log_dir)
        logging.info(f"Log directory is set to: {log_dir}")
    except Exception as e:
        print(f"Failed to create log directory {log_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    logging.info("Starting the third part of the ULTRASAT workflow.")

    # Read parameters from the .ini file first to extract necessary directories
    params = read_params_file(params_file)
    outdir = os.path.abspath(params["save_directory"])
    obs_scenario_dir = os.path.abspath(params["obs_scenario_dir"])

    tiling_time = params["tiling_time"]

    #     # Create all necessary directories
    #     create_directories(outdir)

    # Get the path to the 'ultrasat-gw-followup' directory
    followup_dir = os.path.dirname(params_file)

    # Run compute_tiling.py
    run_compute_tiling(params_file, followup_dir)

    logging.info("compute_tiling.py executed successfully.")

    # Run make-coverage-plots.py
    run_make_coverage_plots(params_file, followup_dir)

    logging.info("make-coverage-plots.py executed successfully.")

    # Optional: Package results for easy download
    # Uncomment the following lines if packaging is desired
    # try:
    #     results_zip = os.path.join(outdir, 'results.zip')
    #     subprocess.run(['zip', '-r', results_zip, os.path.join(outdir, 'coverage_plots'), os.path.join(outdir, 'statistics')], check=True)
    #     logging.info(f"Packaged results into: {results_zip}")
    # except subprocess.CalledProcessError as e:
    #     logging.error(f"Failed to package results: {e.stderr}")

    logging.info("Third part of the ULTRASAT workflow completed successfully.")
    logging.info("Done!")


if __name__ == "__main__":
    main()
