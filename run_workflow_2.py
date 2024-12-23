#!/usr/bin/env python3
"""

Usage:
    python3 run_workflow_2.py --params /path/to/params_file.ini [--log_dir /path/to/logs]

Example:
    python3 run_workflow_2.py --params /home/weizmann.kiendrebeogo/M4OPT-ULTRSASAT/ultrasat-gw-followup/params.ini
"""


import os
import sys
import subprocess
import configparser
import argparse
import logging
from pathlib import Path
import pandas as pd
import textwrap


def setup_logging(log_dir):
    """
    Configure logging for the script.

    Parameters:
        log_dir (str): Directory where log files will be stored.
    """
    log_file = os.path.join(log_dir, "workflow2.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def run_texp_cut_and_batch(params_file, followup_dir):
    """
    Run the texp_cut_and_batch.py script to process exposure times and batch files.

    Parameters:
        params_file (str): Path to the params file (absolute path).
        followup_dir (str): Absolute path to the 'ULTRASAT-followup' directory.
    """
    logging.info(
        "Running texp_cut_and_batch.py to process exposure times and batch files..."
    )
    texp_script = os.path.join(followup_dir, "texp_cut_and_batch.py")

    if not os.path.exists(texp_script):
        logging.error(f"texp_cut_and_batch.py script not found: {texp_script}")
        sys.exit(1)

    try:
        result = subprocess.run(
            ["python3", texp_script, params_file],
            check=True,
            text=True,
            capture_output=True,
        )
        logging.info("texp_cut_and_batch.py completed successfully.")
        logging.debug(f"texp_cut_and_batch.py output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error("Error running texp_cut_and_batch.py:")
        logging.error(e.stderr)
        sys.exit(1)


def create_condor_submission_file_for_scheduler(
    batch_file_path, log_dir, sched_dir, skymap_dir, params
):
    """
    Creates and submits an HTCondor job for each batch file to process ULTRASAT ToO schedules.

    Parameters:
        batch_file_path (str): Path to the batch file.
        log_dir (str): Directory for log files.
        sched_dir (str): Directory for schedule files.
        skymap_dir (str): Directory containing skymap files.
        params (dic): parameters.
    """

    # Determine the path to the dorado-scheduling executable
    try:
        m4opt_path = subprocess.check_output(["which", "m4opt"]).decode().strip()
        if not os.path.exists(m4opt_path):
            raise FileNotFoundError
    except Exception:
        logging.error("m4opt executable not found in PATH.")
        sys.exit(1)

    batch_filename = os.path.basename(batch_file_path).replace(".txt", "")
    job_name = f"ULTRASAT_{batch_filename}"

    try:
        df = pd.read_csv(batch_file_path)
    except Exception as e:
        logging.error(f"Failed to read batch file {batch_file_path}: {e}")
        return

    # read params
    mission = params["mission"]
    deadline = params["deadline"]
    delay = params["delay"]
    job = params["job"]
    nside = params["nside"]

    # Loop through rows and process each event_id and t_exp as required
    for _, row in df.iterrows():
        try:
            event_id = int(row["event_id"])
            texp = row["t_exp (ks)"]
        except (KeyError, ValueError) as e:
            logging.error(f"Invalid data in batch file {batch_file_path}: {e}")
            continue

        logging.info(f"Duration is {deadline}")
        logging.info(f"Processing event {event_id} with exposure time {texp} ks.")

        skymap_file = os.path.join(skymap_dir, f"{event_id}.fits")
        sched_file = os.path.join(sched_dir, f"{event_id}.ecsv")
        wrapper_script = os.path.join(log_dir, f"wrapper_{event_id}.sh")

        wrapper_content = (
            f"#!/bin/bash\n"
            f"\n{m4opt_path} "
            f"schedule "
            f"{skymap_file} "
            f"{sched_file} "
            f"--mission={mission} "
            f"--exptime='{texp} ks' "
            f"--nside={nside} "
            f"--delay='{delay}' "
            f"--deadline='{deadline}' "
            f"--timelimit='20min' "
            f"--jobs {job} "
        )

        # Write the wrapper script
        try:
            with open(wrapper_script, "w") as f:
                f.write(wrapper_content)
            os.chmod(wrapper_script, 0o755)  # Make the script executable
            logging.info(f"Created wrapper script: {wrapper_script}")
        except Exception as e:
            logging.error(f"Failed to create wrapper script for event {event_id}: {e}")
            continue

        # Create the Condor submission script without indentation
        condor_submit_script = textwrap.dedent(
            f"""
            +MaxHours = 24
            universe = vanilla
            accounting_group = ligo.dev.o4.cbc.pe.bayestar
            getenv = true
            executable = {wrapper_script}
            output = {log_dir}/$(Cluster)_$(Process).out
            error = {log_dir}/$(Cluster)_$(Process).err
            log = {log_dir}/$(Cluster)_$(Process).log
            JobBatchName = ULTRASAT_Workflow_{batch_filename}
            request_memory = 60000 MB
            request_disk   = 11000 MB
            request_cpus   = 1
            on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
            on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
            on_exit_hold_reason = (ExitBySignal == True \
                ? strcat("The job exited with signal ", ExitSignal) \
                : strcat("The job exited with code ", ExitCode))
            environment = "OMP_NUM_THREADS=1"
            queue 1
        """
        )

        logging.debug(
            f"Condor submit script for event {event_id}:\n{condor_submit_script}"
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
                logging.info(
                    f"Condor submit output for {batch_filename}, event {event_id}: {stdout.strip()}"
                )
            else:
                logging.error(
                    f"Condor submit error for {batch_filename}, event {event_id}: {stderr.strip()}"
                )
        except Exception as e:
            logging.error(
                f"An error occurred while creating the Condor submission for {batch_filename}, event {event_id}: {e}"
            )


def process_batch_files(batches_dir, log_dir, sched_dir, skymap_dir, params):
    """
    Process all batch files in the specified directory and submit them to HTCondor.

    Parameters:
        batches_dir (str): Directory containing batch files.
        log_dir (str): Directory for log files.
        sched_dir (str): Directory for schedule files.
        skymap_dir (str): Directory containing skymap files.
        params (dic):aprameters.
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
        logging.info(f"Submitting Condor job for batch file: {batch_file_path}")

        create_condor_submission_file_for_scheduler(
            batch_file_path,
            log_dir,
            sched_dir,
            skymap_dir,
            params,
        )


def create_directories(outdir, additional_dirs=None):
    """
    Create necessary directories for the workflow.

    Parameters:
        outdir (str): Base output directory.
        additional_dirs (list, optional): Additional directories to create within outdir.
    """
    directories = [
        outdir,
        os.path.join(outdir, "schedules"),
        os.path.join(outdir, "texp_out"),
        os.path.join(outdir, "texp_sched"),
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
            # 'band': config.get("params", "band"),
            # 'KNe_mag_AB': config.getfloat("params", "KNe_mag_AB"),
            # 'distance_measure': config.get("params", "distance_measure"),
            # 'max_texp': config.getfloat("params", "max_texp"),
            "deadline": config.get("params", "deadline"),
            "mission": config.get("params", "mission"),
            "nside": config.get("params", "nside"),
            # 'skygrid_step' : config.get("params", "skygrid_step"),
            # 'skygrid_method' : config.get("params", "skygrid_method"),
            "delay": config.get("params", "delay"),
            # 'roll_step' : config.get("params", "roll_step"),
            "job": config.get("params", "job"),
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
        description="Run the second part of the ULTRASAT workflow and submit jobs using HTCondor."
    )
    parser.add_argument(
        "-p", "--params", type=str, required=True, help="Path to the params file."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs2",
        help="Directory for log files (default: ./Logs2).",
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

    logging.info("Starting the second part of the ULTRASAT workflow.")

    # Read parameters from the .ini file first to extract necessary directories
    params = read_params_file(params_file)
    outdir = os.path.abspath(params["save_directory"])
    obs_scenario_dir = os.path.abspath(params["obs_scenario_dir"])

    # Create all necessary directories
    create_directories(outdir)

    # Define paths
    sched_dir = os.path.join(outdir, "schedules")
    skymap_dir = os.path.join(obs_scenario_dir, "allsky")

    # Ensure schedules directory exists
    os.makedirs(sched_dir, exist_ok=True)

    # Run the texp_cut_and_batch.py script
    followup_dir = os.path.dirname(params_file)
    run_texp_cut_and_batch(params_file, followup_dir)

    logging.info("texp_cut_and_batch.py executed successfully.")

    # Define the batches directory
    batches_dir = os.path.join(outdir, "texp_sched")

    # Submit Condor jobs for each batch file
    process_batch_files(batches_dir, log_dir, sched_dir, skymap_dir, params)

    logging.info("All batch files processed and submitted as Condor jobs.")


if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# """
# Run the second part of the UVEX workflow and submit jobs using HTCondor.

# Usage:
#     python3 run_workflow_2.py --params /path/to/params_file.ini [--log_dir /path/to/logs]

# Example:
#     python3 run_workflow_2.py --params /home/weizmann.kiendrebeogo/Dorado/uvex-followup/default_params.ini
# """

# import os
# import sys
# import subprocess
# import configparser
# import argparse
# import logging
# from pathlib import Path
# import pandas as pd

# def setup_logging(log_dir):
#     """
#     Configure logging for the script.

#     Parameters:
#         log_dir (str): Directory where log files will be stored.
#     """
#     log_file = os.path.join(log_dir, 'workflow2.log')
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s [%(levelname)s] %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler(sys.stdout)
#         ]
#     )

# def run_texp_cut_and_batch(params_file, followup_dir):
#     """
#     Run the texp_cut_and_batch.py script to process exposure times and batch files.

#     Parameters:
#         params_file (str): Path to the params file (absolute path).
#         followup_dir (str): Absolute path to the 'uvex-followup' directory.
#     """
#     logging.info("Running texp_cut_and_batch.py to process exposure times and batch files...")
#     texp_script = os.path.join(followup_dir, 'texp_cut_and_batch.py')

#     if not os.path.exists(texp_script):
#         logging.error(f"texp_cut_and_batch.py script not found: {texp_script}")
#         sys.exit(1)

#     try:
#         result = subprocess.run(
#             ['python3', texp_script, params_file],
#             check=True,
#             text=True,
#             #capture_output=True
#         )
#         logging.info("texp_cut_and_batch.py completed successfully.")
#         logging.debug(f"texp_cut_and_batch.py output: {result.stdout}")
#     except subprocess.CalledProcessError as e:
#         logging.error("Error running texp_cut_and_batch.py:")
#         logging.error(e.stderr)
#         sys.exit(1)


# def create_condor_submission_file_for_scheduler(batch_file_path, log_dir, sched_dir, skymap_dir, tiling_time):
#     """
#     Creates and submits an HTCondor job for each batch file to process UVEX ToO schedules.

#     Parameters:
#         params (dict): Dictionary containing parameter values.
#         batch_file_path (str): Path to the batch file.
#         log_dir (str): Directory for log files.
#     """

#     # Determine the path to the dorado-scheduling executable
#     try:
#         dorado_scheduling_path = subprocess.check_output(["which", "dorado-scheduling"]).decode().strip()
#         if not os.path.exists(dorado_scheduling_path):
#             raise FileNotFoundError
#     except Exception:
#         logging.error("dorado-scheduling executable not found in PATH.")
#         sys.exit(1)

#     batch_filename = os.path.basename(batch_file_path).replace('.txt', '')
#     job_name = f"UVEX_Scheduler_{batch_filename}"

#     df = pd.read_csv(batch_file_path)

#     # Loop through rows and process each event_id and t_exp as required
#     for _, row in df.iterrows():
#         event_id = row['event_id'].astype('int')
#         texp = row['t_exp (ks)']
#         logging.info(f"Duration is {tiling_time}")
#         logging.info(f"Processing event {event_id} with exposure time {texp} ks.")

#         skymap_file = os.path.join(skymap_dir, f"{event_id}.fits")
#         sched_file = os.path.join(sched_dir, f"{event_id}.ecsv")
#         wrapper_script = os.path.join(log_dir, f"wrapper_{event_id}.sh")

#         wrapper_content = (
#             f"#!/bin/bash\n"
#             f"\n{dorado_scheduling_path} "
#             f"{skymap_file} "
#             f"-o {sched_file} "
#             f"--mission=uvex "
#             f"--exptime='{texp} ks' "
#             f" --duration='{tiling_time}' "
#             f"--roll-step='360 deg' "
#             f"--skygrid-method=sinusoidal "
#             f"--skygrid-step='10 deg2' "
#             f"--nside=128 "
#             f"--delay='10 yr' "
#             f"-j 10"
#         )


#         #arguments = {os.path.join(skymap_dir, f'{event_id}.fits')} -o {os.path.join(sched_dir, f'{event_id}.ecsv')} --mission=uvex --exptime='{texp} ks' --duration='{tiling_time}' --roll-step='360 deg' --skygrid-method=sinusoidal --skygrid-step='10 deg2' --nside=128 --delay='10 yr' -j 10


#         # Écrire le script wrapper
#         try:
#             with open(wrapper_script, 'w') as f:
#                 f.write(wrapper_content)
#             os.chmod(wrapper_script, 0o755)  # Rendre le script exécutable
#             logging.info(f"Created wrapper script: {wrapper_script}")
#         except Exception as e:
#             logging.error(f"Failed to create wrapper script for event {event_id}: {e}")
#             continue


#         condor_submit_script = f'''
#                                 universe = vanilla
#                                 accounting_group = ligo.dev.o4.cbc.pe.bayestar
#                                 getenv = true
#                                 executable = {wrapper_script}
#                                 output = {log_dir}/$(Cluster)_$(Process).out
#                                 error = {log_dir}/$(Cluster)_$(Process).err
#                                 log = {log_dir}/$(Cluster)_$(Process).log
#                                 JobBatchName = UVEX_Workflow_{batch_filename}
#                                 request_memory = 1000 MB
#                                 request_disk   = 100 MB
#                                 request_cpus   = 1
#                                 on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
#                                 on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
#                                 on_exit_hold_reason = (ExitBySignal == True ? "The job exited with signal " . ExitSignal : "The job exited with code " . ExitCode)
#                                 environment = "OMP_NUM_THREADS=1"
#                                 queue
#                             '''
#         print(condor_submit_script)

#         try:
#             proc = subprocess.Popen(
#                 ['condor_submit'],
#                 text=True,
#                 stdin=subprocess.PIPE,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE
#             )
#             stdout, stderr = proc.communicate(input=condor_submit_script)
#             if proc.returncode == 0:
#                 logging.info(f"Condor submit output for {batch_filename}: {stdout.strip()}")
#             else:
#                 logging.error(f"Condor submit error for {batch_filename}: {stderr.strip()}")
#         except Exception as e:
#             logging.error(f"An error occurred while creating the Condor submission for {batch_filename}: {e}")


# def process_batch_files(batches_dir, log_dir, sched_dir, skymap_dir, tiling_time):
#     """
#     Process all batch files in the specified directory and submit them to HTCondor.

#     Parameters:
#         params_file (str): Path to the params file.
#         batches_dir (str): Directory containing batch files.
#         log_dir (str): Directory for log files.
#     """

#     if not os.path.isdir(batches_dir):
#         logging.error(f"Batches directory does not exist: {batches_dir}")
#         sys.exit(1)

#     batch_files = [f for f in os.listdir(batches_dir) if os.path.isfile(os.path.join(batches_dir, f))]
#     if not batch_files:
#         logging.error(f"No batch files found in {batches_dir}. Exiting.")
#         sys.exit(1)


#     for batch_file in batch_files:
#         batch_file_path = os.path.abspath(os.path.join(batches_dir, batch_file))
#         logging.info(f"Submitting Condor job for batch file: {batch_file_path}")

#     create_condor_submission_file_for_scheduler(
#         batch_file_path,
#         log_dir,
#         sched_dir,
#         skymap_dir,
#         tiling_time,
#     )

# def create_directories(outdir, additional_dirs=None):
#     """
#     Create necessary directories for the workflow.

#     Parameters:
#         outdir (str): Base output directory.
#         additional_dirs (list, optional): Additional directories to create within outdir.
#     """
#     directories = [
#         outdir,
#         os.path.join(outdir, 'schedules'),
#         os.path.join(outdir, 'texp_out'),
#         os.path.join(outdir, 'texp_sched')
#     ]
#     if additional_dirs:
#         for additional_dir in additional_dirs:
#             directories.append(os.path.join(outdir, additional_dir))

#     for directory in directories:
#         try:
#             os.makedirs(directory, exist_ok=True)
#             logging.info(f"Ensured directory exists: {directory}")
#         except Exception as e:
#             logging.error(f"Failed to create directory {directory}: {e}")
#             sys.exit(1)

# def read_params_file(params_file):
#     """
#     Read and parse the parameters from the .ini file.

#     Parameters:
#         params_file (str): Path to the params file.

#     Returns:
#         dict: Dictionary containing parameter values.
#     """
#     config = configparser.ConfigParser()
#     config.read(params_file)

#     try:
#         params = {
#             'obs_scenario_dir': config.get("params", "obs_scenario"),
#             'save_directory': config.get("params", "save_directory"),
#             'band': config.get("params", "band"),
#             'KNe_mag_AB': config.getfloat("params", "KNe_mag_AB"),
#             'distance_measure': config.get("params", "distance_measure"),
#             'max_texp': config.getfloat("params", "max_texp"),
#             'tiling_time': config.get("params", "tiling_time")
#         }
#         logging.debug(f"Parameters read from {params_file}: {params}")
#         return params
#     except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
#         logging.error(f"Error reading parameters from {params_file}: {e}")
#         sys.exit(1)


# def parse_arguments():
#     """
#     Parse command-line arguments.

#     Returns:
#         argparse.Namespace: Parsed arguments.
#     """
#     parser = argparse.ArgumentParser(
#         description="Run the second part of the UVEX workflow and submit jobs using HTCondor."
#     )
#     parser.add_argument(
#         "-p", "--params",
#         type=str,
#         required=True,
#         help="Path to the params file."
#     )
#     parser.add_argument(
#         "--log_dir",
#         type=str,
#         default="./Logs2",
#         help="Directory for log files (default: ./logs2)."
#     )
#     return parser.parse_args()


# def main():
#     """
#     Main function to execute the workflow.
#     """
#     args = parse_arguments()

#     # Convert directories to absolute paths
#     log_dir = os.path.abspath(args.log_dir)
#     params_file = os.path.abspath(args.params)

#     # Ensure log directory exists before setting up logging
#     try:
#         os.makedirs(log_dir, exist_ok=True)
#         setup_logging(log_dir)
#         logging.info(f"Log directory is set to: {log_dir}")
#     except Exception as e:
#         print(f"Failed to create log directory {log_dir}: {e}", file=sys.stderr)
#         sys.exit(1)

#     logging.info("Starting the second part of the UVEX workflow.")


#     # Read parameters from the .ini file first to extract necessary directories
#     params = read_params_file(params_file)
#     outdir = os.path.abspath(params['save_directory'])
#     obs_scenario_dir = os.path.abspath(params['obs_scenario_dir'])

#     tiling_time = params['tiling_time']

#     # Create all necessary directories
#     create_directories(outdir)


#     # Define paths
#     sched_dir = os.path.join(outdir, 'schedules')
#     skymap_dir = os.path.join(obs_scenario_dir, 'allsky')

#     # Ensure schedules directory exists
#     os.makedirs(sched_dir, exist_ok=True)

#     # Run the texp_cut_and_batch.py script
#     followup_dir = os.path.dirname(params_file)
#     run_texp_cut_and_batch(params_file, followup_dir)

#     logging.info("texp_cut_and_batch.py executed successfully.")

#     # Define the batches directory
#     batches_dir = os.path.join(outdir, 'texp_sched')

#     # Submit Condor jobs for each batch file
#     process_batch_files(batches_dir, log_dir, sched_dir, skymap_dir, tiling_time)

#     logging.info("All batch files processed and submitted as Condor jobs.")

# if __name__ == "__main__":
#     main()
