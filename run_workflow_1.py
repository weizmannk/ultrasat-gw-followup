# #!/usr/bin/env python3
# """
# Run the ULTRASAT workflow and submit jobs using HTCondor.

# Usage:
#     python3 run_workflow_1.py --params /path/to/params.ini [--log_dir /path/to/logs]
# eg: python3 run_workflow_1.py  --params ./params.ini
# """

import os
import sys
import subprocess
import configparser
import argparse
import logging
import textwrap


def setup_logging(log_dir):
    """
    Configure logging for the script.

    Parameters:
        log_dir (str): Directory where log files will be stored.
    """
    log_file = os.path.join(log_dir, "workflow1.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def run_localization_script(params_file, followup_dir):
    """
    Run the localization script to create batch files and wait for it to finish.

    Parameters:
        params_file (str): Path to the params file (absolute path).
        followup_dir (str): Absolute path to the 'ultrasat-gw-followup' directory.
    """
    logging.info("Running localization script to create batch files...")
    localization_script = os.path.join(followup_dir, "localization_cut_and_batch.py")

    try:
        result = subprocess.run(
            ["python3", localization_script, params_file], check=True, text=True
        )
        logging.info("Localization script completed successfully.")
        logging.debug(f"Localization script output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error("Error running localization script:")
        logging.error(e.stderr)
        sys.exit(1)


def create_condor_submission_file_for_batch(
    script_path, params_file, batch_file, log_dir
):
    """
    Creates and submits an HTCondor job for each batch file.

    Parameters:
        script_path (str): Path to the Python script to execute.
        params_file (str): Path to the params file.
        batch_file (str): Path to the batch file.
        log_dir (str): Directory for log files.
    """

    batch_filename = os.path.basename(batch_file).replace(".txt", "")
    arguments = (
        f"python3 -u {script_path} --params {params_file} --batch_file {batch_file}"
    )

    condor_submit_script = textwrap.dedent(
        f"""
                                +MaxHours = 24
                                universe = vanilla
                                executable =  /usr/bin/env
                                arguments = {arguments}
                                getenv = true
                                request_memory = 10 GB
                                request_disk   = 2  GB
                                request_cpus   = 1
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
        if proc.returncode == 0:
            logging.info(f"Condor submit output for {batch_filename}: {stdout.strip()}")
        else:
            logging.error(f"Condor submit error for {batch_filename}: {stderr.strip()}")
    except Exception as e:
        logging.error(
            f"An error occurred while creating the Condor submission for {batch_filename}: {e}"
        )


def process_batch_files(params_file, followup_dir, batches_dir, log_dir):

    """
    Process all batch files in the specified directory and submit them to HTCondor.

    Parameters:
        params_file (str): Path to the params file.
        followup_dir (str): Path to the 'ultrasat-gw-followup' directory.
        batches_dir (str): Directory containing batch files.
        log_dir (str): Directory for log files.
    """

    if not os.path.isdir(batches_dir):
        logging.error(f"Batches directory does not exist: {batches_dir}")
        sys.exit(1)

    batch_files = [
        f
        for f in os.listdir(batches_dir)
        if os.path.isfile(os.path.join(batches_dir, f))
    ]
    if not batch_files:
        logging.error(f"No batch files found in {batches_dir}. Exiting.")
        sys.exit(1)

    for batch_file in batch_files:
        batch_file_path = os.path.abspath(os.path.join(batches_dir, batch_file))
        logging.info(f"Creating Condor submission for batch file: {batch_file_path}")

        script_path = os.path.abspath(
            os.path.join(followup_dir, "max-texp-by-sky-loc.py")
        )
        abs_params_file = os.path.abspath(params_file)

        if not os.path.exists(script_path):
            logging.error(f"Script not found: {script_path}")
            continue

        create_condor_submission_file_for_batch(
            script_path,
            abs_params_file,
            batch_file_path,
            log_dir,
        )


def create_directories(outdir):
    """
    Create necessary directories for the workflow.

    Parameters:
        outdir (str): Base output directory.
    """
    directories = [
        outdir,
        os.path.join(outdir, "batches"),
        os.path.join(outdir, "texp_out"),
    ]
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
            "KNe_mag_AB": config.getfloat("params", "KNe_mag_AB"),
            "distance_measure": config.get("params", "distance_measure"),
            "max_texp": config.getfloat("params", "max_texp"),
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
        description="Run the ULTRASAT workflow and submit it using HTCondor."
    )
    parser.add_argument(
        "-p", "--params", type=str, required=True, help="Path to the params file."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs1_O5",
        help="Directory for log files (default: ./Logs1).",
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

    # Ensure log directory exists
    try:
        os.makedirs(log_dir, exist_ok=True)
        # Setup logging
        setup_logging(log_dir)
        logging.info(f"Log directory is set to: {log_dir}")

    except Exception as e:
        print(f"Failed to create log directory {log_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    logging.info("Starting ULTRASAT workflow-1.")

    # Read parameters from the .ini file
    params = read_params_file(params_file)
    outdir = os.path.abspath(params["save_directory"])

    # Create all necessary directories
    create_directories(outdir)

    # Run the localization script and process the batch files
    followup_dir = os.path.dirname(params_file)
    run_localization_script(params_file, followup_dir)

    batches_dir = os.path.join(outdir, "batches")
    process_batch_files(params_file, followup_dir, batches_dir, log_dir)

    logging.info("All batch files processed and submitted as Condor jobs.")


if __name__ == "__main__":
    main()
