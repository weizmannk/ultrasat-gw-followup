"""Run the scheduling wrappers on a Dask cluster backed by HTCondor.

Same wrappers as the other backends: the work is shipped as a shell script,
not as an in-process call, so a plan does not depend on the backend.
"""

import logging

from dask_jobqueue import HTCondorCluster
from distributed import Client, as_completed
from tqdm.auto import tqdm

from ultrasat_gw_followup.backend.condor import DEFAULTS
from ultrasat_gw_followup.backend.parallel import run_wrapper

log = logging.getLogger(__name__)


def run_dask(wrapper_scripts, log_dir, settings=None, minimum=1, maximum=20):
    """Run every wrapper as a Dask task on an adaptive HTCondor cluster.

    Parameters
    ----------
    wrapper_scripts : sequence of pathlib.Path
        Scripts to run.
    log_dir : str or pathlib.Path
        Directory the worker logs go to.
    settings : dict, optional
        The ``[condor]`` section of the run configuration; the accounting
        group and the memory and disk requests are taken from it.
    minimum, maximum : int, optional
        Bounds of the adaptive worker pool.

    Returns
    -------
    list of int
        Exit status of each script, in completion order.
    """
    values = DEFAULTS | dict(settings or {})
    cluster = HTCondorCluster(
        cores=1,
        memory=values["request_memory"].replace(" ", ""),
        disk=values["request_disk"].replace(" ", ""),
        log_directory=str(log_dir),
        # dask-jobqueue renamed job_extra to job_extra_directives; the old
        # name is removed in current releases.
        job_extra_directives={"accounting_group": values["accounting_group"]},
        job_script_prologue=["export OMP_NUM_THREADS=1"],
    )
    cluster.adapt(minimum=minimum, maximum=maximum)

    with Client(cluster) as client:
        futures = client.map(run_wrapper, [str(path) for path in wrapper_scripts])
        return [
            future.result()
            for future in tqdm(as_completed(futures), total=len(futures))
        ]
