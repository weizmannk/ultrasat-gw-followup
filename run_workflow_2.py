#!/usr/bin/env python3
"""
ULTRASAT Workflow Execution Script

This script automates the workflow for scheduling and processing ULTRASAT follow-up observations based on gravitational wave (GW) events.

Key Features:
1. Reads observation parameters from a `.ini` configuration file.
2. Executes `./workflow/texp_cut_and_batch.py` to preprocess exposure times and batch files.
3. Iterates over batch files and generates ULTRASAT observation schedules.
4. Submits jobs to HTCondor for scheduling and execution.
5. Logs all processes for monitoring and debugging.

Assumptions:
- Event batch files contain required fields for observation scheduling.
- Exposure time constraints align with ULTRASAT mission parameters.
- `m4opt` is installed and available in the system path.

Usage:
    python3 run_workflow_2.py --params /path/to/params_file.ini [--log_dir /path/to/logs]

Example:
    python3 run_workflow_2.py --params ./params-O5.ini
"""

import os
import sys
import subprocess
import configparser
import argparse
import logging
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

from m4opt.utils.console import status


def setup_logging(log_dir):
    """
    Configure logging for the script.
    """
    log_file = os.path.join(log_dir, "workflow.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def run_texp_cut_and_batch(params_file, followup_dir):
    """
    Run the texp_cut_and_batch.py script to process exposure times and batch files.

    Parameters:
        params_file (str): Path to the params file (absolute path).
        followup_dir (str): Absolute path to the 'ULTRASAT-followup' directory.
    """
    with status("Running exposure times cutoff and batch files..."):
        texp_script = os.path.join(followup_dir, "workflow/texp_cut_and_batch.py")

        if not os.path.exists(texp_script):
            logging.error(
                f"./workflow/texp_cut_and_batch.py script not found: {texp_script}"
            )
            sys.exit(1)

        try:
            result = subprocess.run(
                ["python3", texp_script, params_file],
                check=True,
                text=True,
                capture_output=True,
            )
            logging.info("Exposure times cutoff completed successfully.")
            logging.debug(f"./workflow/texp_cut_and_batch.py output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error("Error running ./workflow/texp_cut_and_batch.py:")
            logging.error(e.stderr)
            sys.exit(1)


def process_batch_files(
    batches_dir,
    log_dir,
    sched_dir,
    skymap_dir,
    prog_dir,
    params,
    number_of_cores,
    parallel=True,
):
    """
    Process all batch files and submit them to HTCondor.
    """
    if not os.path.isdir(batches_dir):
        logging.error(f"Batches directory does not exist: {batches_dir}")
        sys.exit(1)

    if parallel:
        with status("Submit the job on parallele nodes"):
            commands = paralle_process(
                batches_dir, log_dir, sched_dir, skymap_dir, prog_dir, params
            )

            # nodes submission process
            print("Number of jobs remaining... %d." % len(commands))
            parallel_run(commands, number_of_cores)

    else:
        for batch_file in os.listdir(batches_dir):
            with status(f"Submitting Condor job for {batch_file}"):
                batch_file_path = os.path.join(batches_dir, batch_file)
                create_condor_submission(
                    batch_file_path, log_dir, sched_dir, skymap_dir, prog_dir, params
                )


# Parallel nodes submission
def parallel_run(commands, number_of_cores=1):
    Parallel(n_jobs=number_of_cores)(
        delayed(os.system)(command) for command in commands
    )


# Parallel nodes submission
def paralle_process(batches_dir, log_dir, sched_dir, skymap_dir, prog_dir, params):

    try:
        m4opt_path = subprocess.check_output(["which", "m4opt"]).decode().strip()
        if not os.path.exists(m4opt_path):
            raise FileNotFoundError("m4opt executable not found.")
    except Exception:
        logging.error("m4opt executable not found in PATH.")
        sys.exit(1)

    # read params
    mission = params["mission"]
    absmag_mean = params["absmag_mean"]
    absmag_stdev = params["absmag_stdev"]
    snr = params["snr"]
    deadline = params["deadline"]
    delay = params["delay"]
    exptime_min = params["min_texp"]
    max_texp = params["max_texp"]
    bandpass = params["bandpass"].upper()
    nside = params["nside"]
    job = params["job"]

    commands = []
    for batch_file in os.listdir(batches_dir):
        batch_file_path = os.path.join(batches_dir, batch_file)

        try:
            m4opt_path = subprocess.check_output(["which", "m4opt"]).decode().strip()
            if not os.path.exists(m4opt_path):
                raise FileNotFoundError("m4opt executable not found.")
        except Exception:
            logging.error("m4opt executable not found in PATH.")
            sys.exit(1)

        try:
            df = pd.read_csv(batch_file_path)
        except Exception as e:
            logging.error(f"Failed to read batch file {batch_file_path}: {e}")
            return

        for _, row in df.iterrows():
            try:
                event_id = int(row["event_id"])
                # texp = row['t_exp (ks)']
            except (KeyError, ValueError) as e:
                logging.error(f"Invalid data in batch file {batch_file_path}: {e}")
                continue

            skymap_file = os.path.join(skymap_dir, f"{event_id}.fits")
            sched_file = os.path.join(sched_dir, f"{event_id}.ecsv")
            prog_file = os.path.join(prog_dir, f"PROGRESS_{event_id}.ecsv")
            wrapper_script = os.path.join(log_dir, f"wrapper_{event_id}.sh")

            wrapper_content = (
                f"#!/bin/bash\n"
                f"\n{m4opt_path} "
                f"schedule "
                f"{skymap_file} "
                f"{sched_file} "
                f"--mission={mission} "
                f"--bandpass={bandpass} "
                f"--absmag-mean={absmag_mean} "
                f"--absmag-stdev={absmag_stdev} "
                f"--exptime-min='{exptime_min} s' "
                f"--exptime-max='{max_texp} s' "
                f"--snr={snr} "
                f"--delay='{delay}' "
                f"--deadline='{deadline}' "
                f"--timelimit='20min' "
                f"--nside={nside} "
                f"--write-progress {prog_file} "
                f"--jobs {job} "
            )
            commands.append(wrapper_content)

            try:
                with open(wrapper_script, "w") as f:
                    f.write(wrapper_content)
                os.chmod(wrapper_script, 0o755)
            except Exception as e:
                logging.error(
                    f"Failed to create wrapper script for event {event_id}: {e}"
                )
                continue

    return commands


def create_condor_submission(
    batch_file_path, log_dir, sched_dir, skymap_dir, prog_dir, params
):
    """
    Create and submit an HTCondor job for each batch file.
    """
    try:
        m4opt_path = subprocess.check_output(["which", "m4opt"]).decode().strip()
        if not os.path.exists(m4opt_path):
            raise FileNotFoundError("m4opt executable not found.")
    except Exception:
        logging.error("m4opt executable not found in PATH.")
        sys.exit(1)

    batch_filename = os.path.basename(batch_file_path).replace(".csv", "")
    job_name = f"ULTRASAT-Workflow-{batch_filename}"

    # read params
    mission = params["mission"]
    absmag_mean = params["absmag_mean"]
    absmag_stdev = params["absmag_stdev"]
    snr = params["snr"]
    deadline = params["deadline"]
    delay = params["delay"]
    exptime_min = params["min_texp"]
    max_texp = params["max_texp"]
    bandpass = params["bandpass"].upper()
    nside = params["nside"]
    job = params["job"]

    try:
        df = pd.read_csv(batch_file_path)
    except Exception as e:
        logging.error(f"Failed to read batch file {batch_file_path}: {e}")
        return

    for _, row in df.iterrows():
        try:
            event_id = int(row["event_id"])
            # texp = row['t_exp (ks)']
        except (KeyError, ValueError) as e:
            logging.error(f"Invalid data in batch file {batch_file_path}: {e}")
            continue

        skymap_file = os.path.join(skymap_dir, f"{event_id}.fits")
        sched_file = os.path.join(sched_dir, f"{event_id}.ecsv")
        prog_file = os.path.join(prog_dir, f"PROGRESS_{event_id}.ecsv")
        wrapper_script = os.path.join(log_dir, f"wrapper_{event_id}.sh")

        wrapper_content = (
            f"#!/bin/bash\n"
            f"\n{m4opt_path} "
            f"schedule "
            f"{skymap_file} "
            f"{sched_file} "
            f"--mission={mission} "
            f"--bandpass={bandpass} "
            f"--absmag-mean={absmag_mean} "
            f"--absmag-stdev={absmag_stdev} "
            f"--exptime-min='{exptime_min} s' "
            f"--exptime-max='{max_texp} s' "
            f"--snr={snr} "
            f"--delay='{delay}' "
            f"--deadline='{deadline}' "
            f"--timelimit='20min' "
            f"--nside={nside} "
            f"--write-progress {prog_file} "
            f"--jobs {job} "
        )

        try:
            with open(wrapper_script, "w") as f:
                f.write(wrapper_content)
            os.chmod(wrapper_script, 0o755)
        except Exception as e:
            logging.error(f"Failed to create wrapper script for event {event_id}: {e}")
            continue

        condor_submit_script = f"""
        +MaxHours = 24
        universe = vanilla
        accounting_group = ligo.dev.o4.cbc.pe.bayestar
        getenv = true
        executable = {wrapper_script}
        output = {log_dir}/$(Cluster)_$(Process).out
        error = {log_dir}/$(Cluster)_$(Process).err
        log = {log_dir}/$(Cluster)_$(Process).log
        request_memory = 50000 MB
        request_disk = 8000 MB
        request_cpus = 1
        on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
        on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
        on_exit_hold_reason = (ExitBySignal == True \
            ? strcat("The job exited with signal ", ExitSignal) \
            : strcat("The job exited with code ", ExitCode))
        environment = "OMP_NUM_THREADS=1"
        queue 1
        """

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
                logging.error(f"Condor submit error for {event_id}: {stderr.strip()}")
        except Exception as e:
            logging.error(f"Error submitting Condor job for {event_id}: {e}")


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
            "absmag_mean": config.getfloat("params", "absmag_mean"),
            "absmag_stdev": config.getfloat("params", "absmag_stdev"),
            "snr": config.get("params", "snr"),
            "deadline": config.get("params", "deadline"),
            "mission": config.get("params", "mission"),
            "delay": config.get("params", "delay"),
            "min_texp": config.get("params", "min_texp"),
            "max_texp": config.getfloat("params", "max_texp"),
            "bandpass": config.get("params", "bandpass"),
            "nside": config.getint("params", "nside"),
            "job": config.get("params", "job"),
            "dist_measure": config.get("params", "distance_measure"),
            "max_area": config.getfloat("params", "max_area", fallback=2000),
            "number_of_cores": config.getint(
                "params", "number_of_cores", fallback=True
            ),
            "parallel": config.getboolean("params", "parallel", fallback=False),
        }

        logging.debug(f"Parameters read from {params_file}: {params}")
        return params
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        logging.error(f"Error reading parameters from {params_file}: {e}")
        sys.exit(1)


def main():
    """
    Main function to execute the ULTRASAT workflow.
    """
    parser = argparse.ArgumentParser(
        description="Run the ULTRASAT workflow and submit jobs using HTCondor."
    )
    parser.add_argument(
        "-p", "--params", type=str, required=True, help="Path to the params file."
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs2", help="Directory for log files."
    )
    args = parser.parse_args()
    params_file = os.path.abspath(args.params)

    params = read_params_file(params_file)
    number_of_cores = params["number_of_cores"]
    parallel = params["parallel"]

    obs_scenario_dir = os.path.abspath(params["obs_scenario_dir"])
    outdir = os.path.abspath(params["save_directory"])
    skymap_dir = os.path.join(obs_scenario_dir, "allsky")

    log_dir = os.path.join(outdir, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir)

    followup_dir = os.path.dirname(params_file)
    run_texp_cut_and_batch(params_file, followup_dir)

    with status("Schedule Plan"):
        directories = {
            name: os.path.join(outdir, name)
            for name in ["schedules", "progress", "texp_sched"]
        }
        for path in directories.values():
            os.makedirs(path, exist_ok=True)

        sched_dir, prog_dir, batches_dir = directories.values()

        process_batch_files(
            batches_dir,
            log_dir,
            sched_dir,
            skymap_dir,
            prog_dir,
            params,
            number_of_cores,
            parallel,
        )

    if parallel:
        logging.info("All batch files processed in parallel nodes.")

    else:
        logging.info("All batch files processed and submitted as Condor jobs.")


if __name__ == "__main__":
    main()
