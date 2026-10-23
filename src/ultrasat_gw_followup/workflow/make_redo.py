"""Turn the crashed events of a run back into an event table to re-schedule.

Reads the ``crashed.txt`` written by
:mod:`~ultrasat_gw_followup.workflow.check_runs`, selects those rows of the
event table and writes them to ``<out_dir>/redo.ecsv``, then prints the
scheduler command that consumes it.

Usage
-----
::

    python -m ultrasat_gw_followup.workflow.make_redo --config config/uvex.ini
    ugf-make-redo --config config/uvex.ini --first 50

Output: ``<out_dir>/redo.ecsv``.
"""

import argparse
import logging
import shlex
import sys
from pathlib import Path

import numpy as np
from astropy.table import QTable

from ultrasat_gw_followup.config import Config, resolve_input
from ultrasat_gw_followup.workflow.check_runs import read_crashed

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
        help="directory holding crashed.txt and receiving redo.ecsv",
    )
    parser.add_argument(
        "--event-table",
        default=config.get("paths", "event_table", "observing-scenarios.ecsv"),
        help="event table; read from --data-dir unless it is a path of its own",
    )
    parser.add_argument(
        "--first",
        type=int,
        help="keep only the first N crashed events",
    )
    args = parser.parse_args(remaining)
    args.config = config.path
    return args


def select(table, crashed):
    """Return the rows of ``table`` named by a list of crashed events.

    Parameters
    ----------
    table : astropy.table.QTable
        Full event table.
    crashed : sequence of tuple
        ``(run, coinc_event_id)`` pairs, as strings.

    Returns
    -------
    astropy.table.QTable
        The selected rows, in the order of ``table``.

    Raises
    ------
    KeyError
        If a crashed event is absent from the table.
    """
    keys = np.array(
        [
            f"{run}:{event_id}"
            for run, event_id in zip(table["run"], table["coinc_event_id"])
        ]
    )
    wanted = {f"{run}:{event_id}" for run, event_id in crashed}
    missing = wanted - set(keys)
    if missing:
        raise KeyError(
            f"{len(missing)} crashed events absent from the table, e.g. {min(missing)}"
        )
    return table[np.isin(keys, list(wanted))]


def main(argv=None):
    """Entry point of ``ugf-make-redo``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_arguments(argv)

    crashed_file = args.out_dir / "crashed.txt"
    if not crashed_file.exists():
        log.error("%s not found; run ugf-check-runs first", crashed_file)
        return 1

    crashed = read_crashed(crashed_file)
    if args.first is not None:
        crashed = crashed[: args.first]
    if not crashed:
        print(f"{crashed_file} lists no event; nothing to re-run")
        return 0

    table = QTable.read(resolve_input(args.event_table, args.data_dir))
    redo = select(table, crashed)

    path = args.out_dir / "redo.ecsv"
    redo.write(path, overwrite=True)
    config = shlex.quote(str(args.config)) if args.config else "config/<run>.ini"
    print(f"Wrote {path} ({len(redo)} events). Re-run them with:")
    print(f"  ugf-scheduler --config {config} --event-table {shlex.quote(str(path))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
