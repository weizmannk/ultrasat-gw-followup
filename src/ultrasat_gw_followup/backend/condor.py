"""Submit one scheduling wrapper per event to HTCondor.

The memory and disk requests are ClassAd expressions rather than constants:
a job held for exceeding either limit is released and retried with a larger
request, up to a ceiling. Starting low keeps the job matchable on many slots
and lets only the events that genuinely need a large machine escalate to one.
"""

import logging
import re
import subprocess
import textwrap

#: Size suffixes, normalised to KiB. HTCondor reads a bare ``request_memory``
#: as MiB and a bare ``request_disk`` as KiB, so everything is converted to
#: one unit here and re-scaled at the point of use.
_UNITS = {"": 1, "KB": 1, "K": 1, "MB": 1024, "M": 1024, "GB": 1024**2, "G": 1024**2}

#: Hold reason codes that an escalating retry can fix: 34 is the memory
#: limit, 104 the disk limit, and (3, 42) is the wrapper's own exit code for
#: a CPLEX 1803 licensing failure, surfaced through on_exit_hold_subcode.
HOLD_MEMORY = 34
HOLD_DISK = 104
HOLD_POLICY = 3
HOLD_SUBCODE_CPLEX = 42

DEFAULTS = {
    "accounting_group": "ligo.dev.o4.cbc.pe.bayestar",
    "max_hours": "24",
    "request_memory": "40000 MB",
    "request_disk": "20000 MB",
    "max_request_memory": "200000 MB",
    "max_request_disk": "100000 MB",
    "max_retries": "3",
    "growth_factor": "2.0",
}

log = logging.getLogger(__name__)


def to_kb(value):
    """Convert a Condor-style size to KiB.

    Parameters
    ----------
    value : str
        Size such as ``"80000 MB"``, ``"8 GB"`` or a bare ``"4096"``.

    Returns
    -------
    int
        The size in KiB.

    Raises
    ------
    ValueError
        If the string is not a number followed by an optional unit.
    """
    match = re.fullmatch(r"\s*([0-9.]+)\s*([KMG]?B?)\s*", str(value), re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot parse size: {value!r}")
    number, unit = match.groups()
    return int(float(number) * _UNITS[unit.upper()])


def submit_description(wrapper_script, log_dir, settings):
    """Build the HTCondor submit description of one scheduling job.

    ``request_memory`` and ``request_disk`` are floors: each attempt
    re-requests ``growth_factor ** NumJobStarts`` times the floor, capped at
    ``max_request_memory`` and ``max_request_disk``. The memory ceiling must
    stay below the RAM of the largest slot of the pool
    (``condor_status -af Memory | sort -n | tail -1``); above it, an
    escalating job stops matching anything and sits idle instead of failing.

    Parameters
    ----------
    wrapper_script : str or pathlib.Path
        Script the job runs.
    log_dir : str or pathlib.Path
        Directory the stdout, stderr and event logs go to.
    settings : dict
        The ``[condor]`` section of the run configuration. Missing keys fall
        back to :data:`DEFAULTS`.

    Returns
    -------
    str
        The submit description, ready for ``condor_submit``.
    """
    values = DEFAULTS | dict(settings)
    growth = float(values["growth_factor"])
    retries = int(values["max_retries"])

    # request_memory is expressed in MiB, request_disk in KiB.
    memory = max(1, to_kb(values["request_memory"]) // 1024)
    memory_cap = max(memory, to_kb(values["max_request_memory"]) // 1024)
    disk = to_kb(values["request_disk"])
    disk_cap = max(disk, to_kb(values["max_request_disk"]))

    releasable = (
        f"(HoldReasonCode == {HOLD_MEMORY} || HoldReasonCode == {HOLD_DISK} || "
        f"(HoldReasonCode == {HOLD_POLICY} && "
        f"HoldReasonSubCode == {HOLD_SUBCODE_CPLEX}))"
    )
    return textwrap.dedent(f"""\
        +MaxHours = {values["max_hours"]}
        universe = vanilla
        accounting_group = {values["accounting_group"]}
        executable = {wrapper_script}
        output = {log_dir}/$(Cluster)_$(Process).out
        error = {log_dir}/$(Cluster)_$(Process).err
        log = {log_dir}/$(Cluster)_$(Process).log
        request_cpus = 1
        request_memory = int(min({memory_cap}, {memory} * pow({growth}, NumJobStarts)))
        request_disk = int(min({disk_cap}, {disk} * pow({growth}, NumJobStarts)))
        periodic_release = {releasable} && (NumJobStarts <= {retries})
        on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
        on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
        on_exit_hold_subcode = ExitCode
        on_exit_hold_reason = (ExitBySignal == True \
            ? strcat("The job exited with signal ", ExitSignal) \
            : strcat("The job exited with code ", ExitCode))
        environment = "OMP_NUM_THREADS=1"
        queue 1
        """)


def submit_condor_job(run, event_id, log_dir, wrapper_script, settings=None):
    """Submit the wrapper of one event to HTCondor.

    Parameters
    ----------
    run : str
        Observing run the event belongs to, used in the log messages.
    event_id : int or str
        ``coinc_event_id`` of the event.
    log_dir : str or pathlib.Path
        Directory the job logs go to.
    wrapper_script : str or pathlib.Path
        Script the job runs.
    settings : dict, optional
        The ``[condor]`` section of the run configuration.
    """
    description = submit_description(wrapper_script, log_dir, settings or {})
    try:
        process = subprocess.run(
            ["condor_submit"],
            input=description,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        log.error("cannot run condor_submit for %s %s: %s", run, event_id, error)
        return

    if process.returncode:
        log.error(
            "condor_submit failed for %s %s: %s", run, event_id, process.stderr.strip()
        )
    else:
        log.info("submitted %s %s", run, event_id)
