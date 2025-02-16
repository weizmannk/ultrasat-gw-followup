#!/usr/bin/env python3
"""
ULTRASAT Workflow Execution Script

Description     :
                This script automates the execution of the ULTRASAT workflow, including:
                1. Parsing configuration parameters from an `.ini` file.
                2. Running a localization filtering script (`localization_cut_and_batch.py`) to generate batch files.
                   - Filters simulated gravitational wave (GW) events based on sky localization area (`max_area`).
                   - Retains events with localization area below the threshold and groups them into batches.
                   - Uses `classify_populations` to categorize CBC events into BNS, NSBH, and BBH.
                3. Processing GW localization maps (`max-texp-by-sky-loc.py`) to determine maximum exposure times.
                   - Reads GW localization maps and extracts sky coverage probabilities.
                   - Computes the maximum exposure time (`texp_max`) within the 90% credible region.
                   - Filters out events exceeding a maximum allowed exposure time (`max_texp`).
                   - Outputs a list of viable events for follow-up observations.
                4. Submitting batch jobs to HTCondor.
                   - Each batch file is processed independently.
                   - Jobs are submitted via an automated HTCondor submission script.

    Usage:
        python3 run_workflow_1.py --params /path/to/params_file.ini [--log_dir /path/to/logs]

    Example:
        python3 run_workflow_1.py --params ./params-O5.ini
"""

import os
import sys
import subprocess
import configparser
import argparse
import logging
import textwrap
from pathlib import Path
from m4opt.utils.console import status


def setup_logging(log_dir):
    """
    Configure logging for the script.
    Logs are saved in the specified directory and printed to the console.
    """
    log_file = os.path.join(log_dir, "workflow.log")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def run_localization_script(params_file, followup_dir):
    """
    Execute `localization_cut_and_batch.py` to filter and batch events for submission.

    - Filters simulated events based on sky localization area (`max_area`).
    - Groups valid events into batch files (`N_batch` events per batch).
    - Ensures only events suitable for ULTRASAT follow-ups are considered.
    """

    localization_script = os.path.join(followup_dir, "localization_cut_and_batch.py")

    if not os.path.exists(localization_script):
        logging.error(f"Localization script not found: {localization_script}")
        sys.exit(1)

    try:
        subprocess.run(
            ["python3", localization_script, params_file],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logging.error("Error running localization script: %s", e)
        sys.exit(1)


def process_batch_files(params_file, followup_dir, batches_dir, log_dir):
    """
    Process each batch file using `max-texp-by-sky-loc.py` and submit jobs to HTCondor.

    - Reads GW localization maps and extracts sky coverage probabilities.
    - Computes the maximum exposure time (`texp_max`) for each event.
    - Filters out events exceeding ULTRASAT’s maximum exposure constraints.
    - Saves filtered event lists for optimized follow-up observations.
    - Submits each batch file as an independent HTCondor job.
    """
    script_path = os.path.join(followup_dir, "max-texp-by-sky-loc.py")

    if not os.path.exists(script_path):
        logging.error(f"Script not found: {script_path}")
        sys.exit(1)

    if not os.path.isdir(batches_dir):
        logging.error(f"Batches directory does not exist: {batches_dir}")
        sys.exit(1)

    for batch_file in os.listdir(batches_dir):
        batch_file_path = os.path.join(batches_dir, batch_file)
        # logging.info(f"Processing batch file: {batch_file_path}")

        create_condor_submission(script_path, params_file, batch_file_path, log_dir)


def create_condor_submission(script_path, params_file, batch_file, log_dir):
    """
    Creates and submits an HTCondor job for a given batch file.
    """

    batch_filename = os.path.basename(batch_file).replace(".csv", "")
    arguments = (
        f"python3 -u {script_path} --params {params_file} --batch_file {batch_file}"
    )

    condor_submit_script = textwrap.dedent(
        f"""
        universe = vanilla
        executable = /usr/bin/env
        arguments = {arguments}
        getenv = true
        request_memory = 10GB
        request_disk = 2GB
        request_cpus = 1
        accounting_group = ligo.dev.o4.cbc.pe.bayestar
        on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
        on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
        on_exit_hold_reason = (ExitBySignal == True ? "The job exited with signal " . ExitSignal : "The job exited with code " . ExitCode)
        notification = never
        JobBatchName = ULTRASAT_Workflow_{batch_filename}
        environment = "OMP_NUM_THREADS=1"
        output = {log_dir}/$(Cluster)_$(Process).out
        error = {log_dir}/$(Cluster)_$(Process).err
        log = {log_dir}/$(Cluster)_$(Process).log
        queue 1
    """
    )

    try:
        proc = subprocess.Popen(
            ["condor_submit"],
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate(input=condor_submit_script)
        if proc.returncode != 0:
            logging.error(f"Condor submit error for {batch_filename}: {stderr.strip()}")
    except Exception as e:
        logging.error(
            f"An error occurred while creating the Condor submission for {batch_filename}: {e}"
        )


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the ULTRASAT workflow and submit it using HTCondor."
    )
    parser.add_argument(
        "-p", "--params", type=str, required=True, help="Path to the params file."
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs1_O5", help="Directory for log files."
    )
    return parser.parse_args()


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
        params = {"save_directory": config.get("params", "save_directory")}
        logging.debug(f"Parameters read from {params_file}: {params}")
        return params
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        logging.error(f"Error reading parameters from {params_file}: {e}")
        sys.exit(1)


def main():
    """
    Main function to execute the ULTRASAT workflow.
    """
    with status("Starting ULTRASAT workflow N°1."):
        args = parse_arguments()
        log_dir = os.path.abspath(args.log_dir)
        params_file = os.path.abspath(args.params)
        os.makedirs(log_dir, exist_ok=True)

        setup_logging(log_dir)

        # Read parameters from the `.ini` file
        params = read_params_file(params_file)

        # Create required directories
        outdir = os.path.abspath(params["save_directory"])
        batches_dir = os.path.join(outdir, "batches")
        os.makedirs(batches_dir, exist_ok=True)

        followup_dir = os.path.dirname(params_file)

        # Execute workflow steps
        with status("Down-selection Cutoff events"):
            run_localization_script(params_file, followup_dir)

        with status(f"HTCondor job submission"):
            process_batch_files(params_file, followup_dir, batches_dir, log_dir)

    logging.info("All batch files processed and submitted as Condor jobs.")


if __name__ == "__main__":
    main()
