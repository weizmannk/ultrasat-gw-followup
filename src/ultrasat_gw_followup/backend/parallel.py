"""Run the scheduling wrappers locally, in parallel processes."""

import logging
import subprocess

from joblib import Parallel, delayed

log = logging.getLogger(__name__)


def run_wrapper(wrapper_script):
    """Run one wrapper script and log its output.

    Parameters
    ----------
    wrapper_script : str or pathlib.Path
        Script to run.

    Returns
    -------
    int
        Exit status of the script.
    """
    # stderr is folded into stdout rather than captured separately, so that
    # the solver log reads in order; capture_output would conflict with it.
    process = subprocess.run(
        ["bash", str(wrapper_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if process.returncode:
        log.error(
            "%s exited %d: %s",
            wrapper_script,
            process.returncode,
            process.stdout.strip(),
        )
    else:
        log.info("finished %s", wrapper_script)
    return process.returncode


def run_parallel(wrapper_scripts, n_cores):
    """Run every wrapper with ``n_cores`` worker processes.

    Parameters
    ----------
    wrapper_scripts : sequence of pathlib.Path
        Scripts to run.
    n_cores : int
        Number of worker processes.

    Returns
    -------
    list of int
        Exit status of each script, in the order given.
    """
    log.info("running %d wrappers on %d cores", len(wrapper_scripts), n_cores)
    return Parallel(n_jobs=n_cores)(
        delayed(run_wrapper)(script) for script in wrapper_scripts
    )
