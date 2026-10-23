"""Aggregate the schedules of a run into one event table.

For every event, the schedule is read, the kilonova detection probability at
the true position is computed, and the solver metadata and the scheduler
arguments are copied into the event table. Events with no schedule get a row
of nulls.

Usage
-----
::

    python -m ultrasat_gw_followup.workflow.postprocess --config config/uvex.ini
    ugf-postprocess --config config/uvex.ini --output-file uvex.ecsv

Inputs: ``<data_dir>/<event_table>`` and ``<out_dir>/<sched_dir>/<run>/*.ecsv``.
Output: ``<out_dir>/<output_file>``.
"""

import argparse
import logging
import sys
from functools import partial
from pathlib import Path

import numpy as np
from astropy.table import QTable
from ligo.skymap.util.progress import progress_map

from ultrasat_gw_followup.config import Config, resolve_input
from ultrasat_gw_followup.probability.detection_probability import (
    get_detection_probability_known_position,
    observed_fields,
)

#: Output columns, in the order :func:`process` returns them.
COLUMNS = (
    "detection_probability_known_position",
    "objective_value",
    "best_bound",
    "solution_status",
    "solution_time",
    "num_fields",
    "field_ids",
    "cutoff",
    "nside",
    "mission",
    "snr",
    "deadline",
    "delay",
    "exptime_min",
    "exptime_max",
    "bandpass",
    "absmag_mean",
    "absmag_stdev",
    "visits",
    "skymap",
    "skygrid",
)

log = logging.getLogger(__name__)


def parse_arguments(argv=None):
    """Parse the command line, with the ``.ini`` values as defaults."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], allow_abbrev=False
    )
    config, remaining = Config.from_command_line(parser, argv)

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(config.get("paths", "data_dir", "data")),
        help="directory holding the event table",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(config.get("paths", "out_dir", "output")),
        help="directory holding the schedules and receiving the result table",
    )
    parser.add_argument(
        "--sched-dir",
        default=config.get("paths", "sched_dir", "schedules"),
        help="schedule directory, relative to --out-dir",
    )
    parser.add_argument(
        "--event-table",
        default=config.get("paths", "event_table", "observing-scenarios.ecsv"),
        help="event table; read from --data-dir unless it is a path of its own",
    )
    parser.add_argument(
        "--output-file",
        default=config.get("paths", "output_file", "events.ecsv"),
        help="result table, written inside --out-dir",
    )
    return parser.parse_args(remaining)


def read_plan_args(plan):
    """Return the scheduler arguments stored in a schedule, normalised.

    m4opt records ``bandpass`` and ``exptime_min`` as one entry per visit.
    This pipeline uses a single bandpass and a single minimum exposure time,
    so the lists are collapsed back to scalars.

    Raises
    ------
    NotImplementedError
        If the plan used more than one bandpass or exposure time.
    """
    plan_args = dict(plan.meta["args"])
    for key in ("bandpass", "exptime_min"):
        value = plan_args.get(key)
        if isinstance(value, list):
            if len({str(item) for item in value}) > 1:
                raise NotImplementedError(f"several values for {key}: {value}")
            plan_args[key] = value[0] if value else None
    return plan_args


def process(row, sched_path):
    """Read the schedule of one event and return its output columns.

    Parameters
    ----------
    row : astropy.table.Row
        Event, read for its run, id, true position and distance.
    sched_path : pathlib.Path
        Directory holding the per-run schedule directories.

    Returns
    -------
    tuple
        One value per entry of :data:`COLUMNS`. Every value is ``None`` when
        the schedule is missing, except ``field_ids``, which stays an empty
        array: a ``None`` cannot be written to a variable-length ECSV column.
    """
    run, event_id = row["run"], row["coinc_event_id"]
    plan_file = sched_path / run / f"{event_id}.ecsv"

    if not plan_file.exists():
        log.warning("missing schedule %s", plan_file)
        return (None,) * 6 + (np.array([], dtype=int),) + (None,) * 14

    plan = QTable.read(plan_file)
    plan_args = read_plan_args(plan)
    plan_args.pop("skymap", None)

    observations = observed_fields(plan)
    if "field_id" in observations.colnames:
        # m4opt names the sky grid tile of every observation, so the fields
        # can be counted directly rather than assuming that each one received
        # all of its visits -- which breaks on plans stopped on the time limit.
        field_ids = np.unique(np.asarray(observations["field_id"]))
        num_fields = len(field_ids)
    else:
        field_ids = np.array([], dtype=int)
        num_fields = len(observations) // plan_args["visits"]

    return (
        get_detection_probability_known_position(plan, row, plan_args),
        plan.meta.get("objective_value"),
        plan.meta.get("best_bound"),
        plan.meta.get("solution_status"),
        plan.meta.get("solution_time"),
        num_fields,
        field_ids,
        plan_args.get("cutoff"),
        plan_args.get("nside"),
        plan_args.get("mission"),
        plan_args.get("snr"),
        plan_args.get("deadline"),
        plan_args.get("delay"),
        plan_args.get("exptime_min"),
        plan_args.get("exptime_max"),
        plan_args.get("bandpass"),
        plan_args.get("absmag_mean"),
        plan_args.get("absmag_stdev"),
        plan_args.get("visits"),
        plan_args.get("skymap"),
        plan_args.get("skygrid"),
    )


def main(argv=None):
    """Entry point of ``ugf-postprocess``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    args = parse_arguments(argv)

    event_table = resolve_input(args.event_table, args.data_dir)
    sched_path = args.out_dir / args.sched_dir
    output_path = args.out_dir / args.output_file

    log.info("reading %s", event_table)
    table = QTable.read(event_table)

    # The scheduler's own "snr" is the kilonova detection threshold and
    # overwrites the network SNR of the GW event below. Keep the latter.
    if "snr" in table.colnames:
        table["network_snr"] = table["snr"]

    log.info("computing detection probabilities and solver metrics")
    # progress_map ships the callable to worker processes through pickle, so
    # it cannot be a closure. It pickles each item too, and an astropy Row
    # keeps a reference to its parent table: detaching each row into its own
    # one-row table first cuts the traffic by two orders of magnitude.
    rows = [table[i : i + 1][0] for i in range(len(table))]
    results = zip(
        *progress_map(partial(process, sched_path=sched_path), rows, jobs=None)
    )
    for name, values in zip(COLUMNS, results):
        table[name] = values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("writing %s", output_path)
    table.write(output_path, overwrite=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
