#!/usr/bin/env python
"""Report which events of a run have a usable schedule, and which do not.

An event counts as crashed when its schedule is missing, cannot be read, or
records the CPLEX 1803 licensing error. Those are written to
``<out_dir>/crashed.txt``, which ``make_redo.py`` turns back into an event
table. Run it when the queue has drained, between scheduling and
post-processing.

Usage
-----
::

    python scripts/check_runs.py --config config/uvex.ini

Output: ``<out_dir>/crashed.txt``, one ``<run> <coinc_event_id>`` per line,
plus a summary of the solution statuses per observing run.
"""

import argparse
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from astropy.table import QTable

from ultrasat_gw_followup.config import Config, resolve_input
from ultrasat_gw_followup.workflow.scheduler import CPLEX_ERROR_PATTERN

CPLEX_ERROR = re.compile(CPLEX_ERROR_PATTERN)

#: Status reported for an event whose schedule is absent or unreadable.
MISSING = "missing"
UNREADABLE = "unreadable"
CPLEX_1803 = "CPLEX Error 1803"

#: Statuses that make an event a candidate for a re-run.
CRASHED = frozenset({MISSING, UNREADABLE, CPLEX_1803})

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
        help="directory holding the schedules and receiving crashed.txt",
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
    args = parser.parse_args(remaining)
    args.config = config.path
    return args


def schedule_status(plan_file):
    """Return the status of one schedule file.

    Parameters
    ----------
    plan_file : pathlib.Path
        Schedule written by ``m4opt schedule``.

    Returns
    -------
    str
        :data:`MISSING`, :data:`UNREADABLE`, :data:`CPLEX_1803`, or the
        ``solution_status`` recorded by the solver.
    """
    if not plan_file.exists():
        return MISSING
    try:
        plan = QTable.read(plan_file)
    except Exception:  # noqa: BLE001 -- any unreadable plan has to be re-run
        return UNREADABLE
    status = str(plan.meta.get("solution_status", ""))
    return CPLEX_1803 if CPLEX_ERROR.search(status) else status


def check(table, sched_path):
    """Collect the status of every event of a table.

    Parameters
    ----------
    table : astropy.table.QTable
        Event table, read for ``run`` and ``coinc_event_id``.
    sched_path : pathlib.Path
        Directory holding the per-run schedule directories.

    Returns
    -------
    statuses : dict
        ``{run: Counter of statuses}``.
    crashed : list of tuple
        ``(run, coinc_event_id)`` of the events that need a re-run.
    """
    statuses = defaultdict(Counter)
    crashed = []
    for row in table:
        run, event_id = str(row["run"]), row["coinc_event_id"]
        status = schedule_status(sched_path / run / f"{event_id}.ecsv")
        statuses[run][status] += 1
        if status in CRASHED:
            crashed.append((run, event_id))
    return dict(statuses), crashed


def write_crashed(path, crashed):
    """Write the crashed events, one ``<run> <coinc_event_id>`` per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{run} {event_id}\n" for run, event_id in crashed))
    return path


def read_crashed(path):
    """Read back a ``crashed.txt`` written by :func:`write_crashed`."""
    lines = (
        line.split() for line in Path(path).read_text().splitlines() if line.strip()
    )
    return [(run, event_id) for run, event_id in lines]


def print_summary(statuses):
    """Print the status counts of each observing run."""
    for run in sorted(statuses):
        counts = statuses[run]
        total = sum(counts.values())
        print(f"{run}  {total} events")
        for status, count in counts.most_common():
            print(f"  {count:6d}  {status or '(no status)'}")


def main(argv=None):
    """Report the statuses of a run and list the events needing a re-run."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_arguments(argv)

    table = QTable.read(resolve_input(args.event_table, args.data_dir))
    statuses, crashed = check(table, args.out_dir / args.sched_dir)

    print_summary(statuses)
    path = write_crashed(args.out_dir / "crashed.txt", crashed)
    print(f"\n{len(crashed)} events to re-run, listed in {path}")
    if crashed:
        where = (
            f"--config {args.config}" if args.config else f"--out-dir {args.out_dir}"
        )
        print(f"Build their event table with: python scripts/make_redo.py {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
