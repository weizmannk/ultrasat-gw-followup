"""Schedule one m4opt observing plan per event, through a batch backend.

For every event of the table this step writes a wrapper shell script that runs
``m4opt schedule``, then hands the wrappers to the chosen backend: HTCondor,
Slurm, local processes, or Dask. All four run the same wrapper, so a plan does
not depend on where it was computed.

Usage
-----
::

    python -m ultrasat_gw_followup.workflow.scheduler --config config/uvex.ini
    ugf-scheduler --config config/uvex.ini --dry-run
    ugf-scheduler --config config/uvex.ini --skip-existing

Inputs: ``<data_dir>/<event_table>`` and ``<data_dir>/<skymap_dir>/<run>/*.fits``.
Outputs: ``<out_dir>/{schedules,progress,logs}/``.
"""

import argparse
import logging
import shlex
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
from astropy.table import QTable
from tqdm.auto import tqdm

from ultrasat_gw_followup.backend.condor import submit_condor_job
from ultrasat_gw_followup.backend.dask import run_dask
from ultrasat_gw_followup.backend.parallel import run_parallel
from ultrasat_gw_followup.backend.slurm import submit_slurm
from ultrasat_gw_followup.config import Config, resolve_input

#: CPLEX reports a licensing failure as error 1803 and m4opt still writes a
#: plan, with the message in ``solution_status``. The wrapper greps for it and
#: exits with this code so that the job is held and retried rather than
#: counted as a success.
CPLEX_ERROR_PATTERN = r"CPLEX Error +1803"
CPLEX_ERROR_EXIT_CODE = 42

BACKENDS = ("condor", "slurm", "parallel", "dask")

log = logging.getLogger(__name__)


def parse_arguments(argv=None):
    """Parse the command line, with the ``.ini`` values as defaults."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], allow_abbrev=False
    )
    config, remaining = Config.from_command_line(parser, argv)

    # [mission]
    parser.add_argument(
        "--mission",
        default=config.get("mission", "mission"),
        required=config.get("mission", "mission") is None,
        help="m4opt mission name",
    )
    parser.add_argument(
        "--skygrid",
        default=config.get("mission", "skygrid"),
        help="sky grid, for missions that have several",
    )
    parser.add_argument(
        "--bandpass",
        default=config.get("mission", "bandpass"),
        required=config.get("mission", "bandpass") is None,
        help="detector bandpass",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=config.getint("mission", "nside", 128),
        help="HEALPix resolution",
    )

    # [kilonova]
    parser.add_argument(
        "--absmag-mean",
        type=float,
        default=config.getfloat("kilonova", "absmag_mean", -16.0),
        help="mean AB absolute magnitude of the kilonova",
    )
    parser.add_argument(
        "--absmag-stdev",
        type=float,
        default=config.getfloat("kilonova", "absmag_stdev", 1.3),
        help="standard deviation of the AB absolute magnitude",
    )

    # [scheduling]. The options left at None are passed through to m4opt only
    # when set, so that its own default applies otherwise.
    parser.add_argument(
        "--snr",
        type=float,
        default=config.getfloat("scheduling", "snr", 5),
        help="signal-to-noise ratio of a kilonova detection",
    )
    parser.add_argument(
        "--delay",
        default=config.get("scheduling", "delay", "15min"),
        help="delay between the alert and the start of observations",
    )
    parser.add_argument(
        "--deadline",
        default=config.get("scheduling", "deadline", "6hour"),
        help="end of the follow-up window, measured from the merger",
    )
    parser.add_argument(
        "--exptime-min",
        type=int,
        default=config.getint("scheduling", "exptime_min", 300),
        help="minimum exposure time, in seconds",
    )
    parser.add_argument(
        "--exptime-max",
        type=float,
        default=config.getfloat("scheduling", "exptime_max", np.inf),
        help="maximum exposure time, in seconds; inf leaves it unconstrained",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=config.getfloat("scheduling", "cutoff"),
        help="objective cutoff below which the scheduler gives up",
    )
    parser.add_argument(
        "--visits",
        type=int,
        default=config.getint("scheduling", "visits"),
        help="number of visits per field",
    )
    parser.add_argument(
        "--cadence",
        default=config.get("scheduling", "cadence"),
        help="minimum separation between visits",
    )
    parser.add_argument(
        "--appmag-dist",
        action=argparse.BooleanOptionalAction,
        default=config.getbool("scheduling", "appmag_dist"),
        help="variable exposure with an absolute-magnitude distribution",
    )
    parser.add_argument(
        "--max-fields",
        type=int,
        default=config.getint("scheduling", "max_fields"),
        help="keep only the N most probable fields",
    )

    # [solver]
    parser.add_argument(
        "--timelimit",
        default=config.get("solver", "timelimit"),
        help="wall-clock limit of one MILP solve",
    )
    parser.add_argument(
        "--memory",
        default=config.get("solver", "memory"),
        help="maximum size of the CPLEX node file",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=config.getint("solver", "jobs", 0),
        help="threads per solve; 0 uses all cores",
    )

    # [paths]
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(config.get("paths", "data_dir", "data")),
        help="directory holding the event table and the sky maps",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(config.get("paths", "out_dir", "output")),
        help="directory the schedules, progress files and logs are written to",
    )
    parser.add_argument(
        "--skymap-dir",
        default=config.get("paths", "skymap_dir", "skymaps"),
        help="sky map directory, relative to --data-dir",
    )
    parser.add_argument(
        "--sched-dir",
        default=config.get("paths", "sched_dir", "schedules"),
        help="schedule directory, relative to --out-dir",
    )
    parser.add_argument(
        "--prog-dir",
        default=config.get("paths", "prog_dir", "progress"),
        help="progress directory, relative to --out-dir",
    )
    parser.add_argument(
        "--log-dir",
        default=config.get("paths", "log_dir", "logs"),
        help="log and wrapper directory, relative to --out-dir",
    )
    parser.add_argument(
        "--event-table",
        default=config.get("paths", "event_table", "observing-scenarios.ecsv"),
        help="event table; read from --data-dir unless it is a path of its own",
    )

    # [execution]
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=config.get("execution", "backend", "condor"),
        help="where the wrappers are run",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=config.getint("execution", "n_cores", 4),
        help="worker processes of the parallel backend",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip events whose schedule file already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the wrappers but submit nothing",
    )

    args = parser.parse_args(remaining)
    args.condor = config.getsection("condor")
    args.slurm = config.getsection("slurm")
    return args


def setup_logging(log_dir):
    """Log to the console and to ``<log_dir>/workflow.log``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(Path(log_dir) / "workflow.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def build_command(executable, skymap_file, sched_file, prog_file, args):
    """Build the ``m4opt schedule`` command line of one event.

    Options that are ``None`` are left out, so that m4opt applies its own
    default and stays correct if that default moves. ``--event-time`` is
    deliberately absent: m4opt reads the trigger time from the ``DATE-OBS``
    header of the sky map.

    Parameters
    ----------
    executable : str
        Path of the ``m4opt`` executable.
    skymap_file, sched_file, prog_file : pathlib.Path
        Input sky map, output schedule, output progress file.
    args : argparse.Namespace
        Parsed options.

    Returns
    -------
    list of str
        The command, one token per element.
    """
    command = [executable, "schedule", str(skymap_file), str(sched_file)]
    command.append(f"--mission={args.mission}")
    if args.skygrid:
        command.append(f"--skygrid={args.skygrid}")
    command += [
        f"--bandpass={args.bandpass}",
        f"--nside={args.nside}",
        f"--absmag-mean={args.absmag_mean}",
        f"--absmag-stdev={args.absmag_stdev}",
        f"--snr={args.snr}",
        f"--delay={args.delay}",
        f"--deadline={args.deadline}",
        f"--exptime-min={args.exptime_min} s",
        f"--exptime-max={args.exptime_max} s",
    ]
    if args.cutoff is not None:
        command.append(f"--cutoff={args.cutoff}")
    if args.visits is not None:
        command.append(f"--visits={args.visits}")
    if args.cadence is not None:
        command.append(f"--cadence={args.cadence}")
    if args.appmag_dist is not None:
        command.append("--appmag-dist" if args.appmag_dist else "--no-appmag-dist")
    if args.max_fields is not None:
        command.append(f"--max-fields={args.max_fields}")
    if args.timelimit is not None:
        command.append(f"--timelimit={args.timelimit}")
    if args.memory is not None:
        command.append(f"--memory={args.memory}")
    command += [f"--jobs={args.jobs}", f"--write-progress={prog_file}"]
    return command


def wrapper_text(command, sched_file):
    """Return the shell script that runs one scheduling command."""
    return f"""#!/bin/bash
# Generated by ultrasat_gw_followup.workflow.scheduler. Do not edit.
export OMP_NUM_THREADS=1
# CPLEX writes its node file to TMPDIR. Under HTCondor the only writable space
# of the slot is the job scratch directory, so prefer it when it is set.
export TMPDIR="${{_CONDOR_SCRATCH_DIR:-${{TMPDIR:-/tmp}}}}"

{shlex.join(command)}
status=$?
if [ "$status" -ne 0 ]; then
    exit "$status"
fi

# CPLEX error 1803 is a licensing failure: m4opt exits 0 and writes a plan
# carrying the error in solution_status. Fail the job so that it is retried.
if grep -qE {shlex.quote(CPLEX_ERROR_PATTERN)} {shlex.quote(str(sched_file))}; then
    exit {CPLEX_ERROR_EXIT_CODE}
fi
"""


def create_wrapper(run, event_id, paths, args, executable):
    """Write the wrapper script of one event and make it executable.

    Parameters
    ----------
    run : str
        Observing run the event belongs to.
    event_id : int or str
        ``coinc_event_id`` of the event.
    paths : dict
        Directories, as returned by :func:`make_directories`.
    args : argparse.Namespace
        Parsed options.
    executable : str
        Path of the ``m4opt`` executable.

    Returns
    -------
    pathlib.Path
        Path of the wrapper script.
    """
    sched_file = paths["sched"] / run / f"{event_id}.ecsv"
    prog_file = paths["prog"] / run / f"PROGRESS_{event_id}.ecsv"
    skymap_file = paths["skymap"] / run / f"{event_id}.fits"
    wrapper = paths["log"] / f"wrapper_{run}_{event_id}.sh"

    command = build_command(executable, skymap_file, sched_file, prog_file, args)
    wrapper.write_text(wrapper_text(command, sched_file))
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


def make_directories(args):
    """Create and return the input and output directories of a run."""
    paths = {
        "skymap": args.data_dir / args.skymap_dir,
        "sched": args.out_dir / args.sched_dir,
        "prog": args.out_dir / args.prog_dir,
        "log": args.out_dir / args.log_dir,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def find_m4opt():
    """Return the path of the ``m4opt`` executable.

    Raises
    ------
    FileNotFoundError
        If it is not on ``PATH``.
    """
    executable = shutil.which("m4opt")
    if executable is None:
        raise FileNotFoundError("m4opt executable not found on PATH")
    return executable


def plan_events(table, paths, args):
    """Return the ``(run, event_id)`` pairs still to be scheduled.

    Events whose sky map is missing are skipped with a warning; with
    ``--skip-existing`` so are those that already have a schedule.
    """
    events = []
    for row in table:
        run, event_id = str(row["run"]), row["coinc_event_id"]
        if not (paths["skymap"] / run / f"{event_id}.fits").exists():
            log.warning("missing sky map for %s %s, skipping", run, event_id)
            continue
        if args.skip_existing and (paths["sched"] / run / f"{event_id}.ecsv").exists():
            continue
        events.append((run, event_id))
    return events


def submit(events, wrappers, paths, args):
    """Hand the wrappers to the backend named by ``args.backend``."""
    if args.backend == "condor":
        for (run, event_id), wrapper in tqdm(
            zip(events, wrappers), total=len(wrappers), desc="Submitting"
        ):
            submit_condor_job(run, event_id, paths["log"], wrapper, args.condor)
    elif args.backend == "slurm":
        submit_slurm(wrappers, paths["log"], args.slurm)
    elif args.backend == "parallel":
        run_parallel(wrappers, args.n_cores)
    else:
        run_dask(wrappers, paths["log"], args.condor)


def main(argv=None):
    """Entry point of ``ugf-scheduler``."""
    args = parse_arguments(argv)
    paths = make_directories(args)
    setup_logging(paths["log"])

    try:
        executable = find_m4opt()
    except FileNotFoundError as error:
        log.error("%s", error)
        return 1

    event_table = resolve_input(args.event_table, args.data_dir)
    try:
        table = QTable.read(event_table)
    except Exception as error:  # noqa: BLE001 -- astropy raises many types here
        log.error("cannot read the event table %s: %s", event_table, error)
        return 1

    events = plan_events(table, paths, args)
    for run in {run for run, _ in events}:
        (paths["sched"] / run).mkdir(parents=True, exist_ok=True)
        (paths["prog"] / run).mkdir(parents=True, exist_ok=True)

    wrappers = [
        create_wrapper(run, event_id, paths, args, executable)
        for run, event_id in events
    ]
    log.info("wrote %d wrappers to %s", len(wrappers), paths["log"])

    if args.dry_run:
        log.info("dry run: nothing submitted")
        return 0
    if not wrappers:
        log.info("nothing to submit")
        return 0

    submit(events, wrappers, paths, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
