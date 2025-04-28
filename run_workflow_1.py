#!/usr/bin/env python3
"""
ULTRASAT Workflow Execution Script

Description     :
                This script automates the execution of the ULTRASAT workflow, including:
                1. Parsing configuration parameters from an `.ini` file.
                2. Running a localization filtering script (`./workflow/localization_cut_and_batch.py`) to generate batch files.
                   - Filters simulated gravitational wave (GW) events based on sky localization area (`max_area`).
                   - Retains events with localization area below the threshold and groups them into batches.
                   - Uses `classify_populations` to categorize CBC events into BNS, NSBH, and BBH.
                3. Processing GW localization maps (`./workflow/max-texp-by-sky-loc.py`) to determine maximum exposure times.
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
from joblib import Parallel, delayed
import shutil

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


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

    localization_script = os.path.join(
        followup_dir, "workflow/localization_cut_and_batch.py"
    )

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


def process_batch_files(
    params_file, followup_dir, batches_dir, log_dir, number_of_cores, backend="parallel"
):
    """
    Process each batch file using `./workflow/max-texp-by-sky-loc.py` and submit jobs to HTCondor.

    - Reads GW localization maps and extracts sky coverage probabilities.
    - Computes the maximum exposure time (`texp_max`) for each event.
    - Filters out events exceeding ULTRASAT’s maximum exposure constraints.
    - Saves filtered event lists for optimized follow-up observations.
    - Submits each batch file as an independent HTCondor job.
    """
    script_path = os.path.join(followup_dir, "workflow/max-texp-by-sky-loc.py")

    if not os.path.exists(script_path):
        logging.error(f"Script not found: {script_path}")
        sys.exit(1)

    if not os.path.isdir(batches_dir):
        logging.error(f"Batches directory does not exist: {batches_dir}")
        sys.exit(1)

    if backend == "parallel":
        print("Submit the job on parallele nodes")
        commands = []
        for batch_file in os.listdir(batches_dir):
            batch_file_path = os.path.join(batches_dir, batch_file)
            command = f"python3 -u {script_path} --params {params_file} --batch_file {batch_file_path}"
            print(f"Command: {command}")
            commands.append(command)

        # Parallel nodes submission process
        parallel_run(commands, number_of_cores)

    elif backend == "condor":
        print(f"HTCondor job submission")
        for batch_file in os.listdir(batches_dir):
            batch_file_path = os.path.join(batches_dir, batch_file)
            # condor submission process
            create_condor_submission(script_path, params_file, batch_file_path, log_dir)

    elif backend == "slurm":
        if shutil.which("sbatch") is None:
            logging.warning("SLURM not available on this cluster. Skipping SLURM submission.")
            return
        else:
            print("SLURM job submission")
            batch_files = [
                os.path.join(batches_dir, bf)
                for bf in sorted(os.listdir(batches_dir))
                if bf.endswith(".csv")
            ]
            create_slurm_submission(script_path, params_file, batch_files, log_dir)

    else:
        print("Unknown backend: use 'condor', 'parallel', or 'dask'")
        sys.exit(1)

# SLURM job submission
def create_slurm_submission(script_path, params_file, batch_files, log_dir):
    array_length = len(batch_files)
    batch_file_list_path = os.path.join(log_dir, "batch_files_list.txt")

    # Write batch file paths to a text file
    with open(batch_file_list_path, "w") as f:
        for bf in batch_files:
            f.write(bf + "\n")

    slurm_script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=ULTRASAT_array
        #SBATCH --output={log_dir}/array_%A_%a.out
        #SBATCH --error={log_dir}/array_%A_%a.err
        #SBATCH --partition=cpu
        #SBATCH --account=bcrv-delta-cpu
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=1
        #SBATCH --mem=10G
        #SBATCH --mail-type=FAIL
        #SBATCH --mail-user=leggi014@umn.edu
        #SBATCH --time=12:00:00
        #SBATCH --array=0-{array_length - 1}

        export OMP_NUM_THREADS=1

        batch_file=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {batch_file_list_path})
        echo "Running batch file: $batch_file"

        python3 -u {script_path} --params {params_file} --batch_file "$batch_file"
    """)

    slurm_array_path = os.path.join(log_dir, "slurm_array.sh")
    with open(slurm_array_path, "w") as f:
        f.write(slurm_script)

    try:
        subprocess.run(["sbatch", slurm_array_path], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"SLURM array submission failed: {e}")

# Parallel nodes submission
def parallel_run(commands, number_of_cores=1):
    print(f"Submit the job on {number_of_cores} nodes")
    Parallel(n_jobs=number_of_cores)(
        delayed(os.system)(command) for command in commands
    )


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
        "--log_dir", type=str, default="./logs1", help="Directory for log files."
    )
    return parser.parse_args()


def main():
    """
    Main function to execute the ULTRASAT workflow.
    """
    print("Starting ULTRASAT workflow N°1.")
    args = parse_arguments()

    params_file = os.path.abspath(args.params)
    config = configparser.ConfigParser()
    config.read(params_file)

    
    number_of_cores = config.getint("params", "number_of_cores", fallback=True)
    
    # Chech the backend submission method
    backend = config.get("params", "backend", fallback="parallel")

    # Create required directories
    outdir = os.path.abspath(config.get("params", "save_directory"))
    batches_dir = os.path.join(outdir, "batches")
    os.makedirs(batches_dir, exist_ok=True)

    log_dir = os.path.join(outdir, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir)

    followup_dir = os.path.dirname(params_file)

    # Execute workflow steps
    print("Down-selection Cutoff events")
    run_localization_script(params_file, followup_dir)

    process_batch_files(
        params_file, followup_dir, batches_dir, log_dir, number_of_cores, backend
    )

    if backend == "parallel":
        logging.info("All batch files processed in parallel.")
    elif backend == "slurm":
        logging.info("All batch files processed in SLURM.")
    elif backend == "condor":
        logging.info("All batch files processed in HT Condor.")
    else:
        logging.error("Unknown backend specified in the config file.")


if __name__ == "__main__":
    main()
