"""Run the scheduling wrappers as Slurm array jobs.

Every directive comes from the ``[slurm]`` section of the run configuration;
nothing is hard-coded here, so the same code works on any cluster. The
wrappers are split into arrays of at most ``max_array`` tasks, because
schedulers cap the size of a single array.
"""

import logging
import subprocess
from pathlib import Path

#: Configuration key -> Slurm directive, in the order they are written.
DIRECTIVES = (
    ("partition", "--partition"),
    ("account", "--account"),
    ("cpus", "--cpus-per-task"),
    ("mem", "--mem"),
    ("time", "--time"),
)

log = logging.getLogger(__name__)


def array_script(wrapper_list, log_dir, settings, chunk, n_tasks, job_name):
    """Build the batch script of one array job.

    Parameters
    ----------
    wrapper_list : pathlib.Path
        File listing one wrapper path per line; the array index selects a line.
    log_dir : pathlib.Path
        Directory the Slurm logs go to.
    settings : dict
        The ``[slurm]`` section of the run configuration.
    chunk : int
        Index of the first wrapper of this array, inside ``wrapper_list``.
    n_tasks : int
        Number of tasks in this array.
    job_name : str
        Name given to the job.

    Returns
    -------
    str
        The batch script.
    """
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir}/slurm_%A_%a.out",
        f"#SBATCH --error={log_dir}/slurm_%A_%a.err",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
    ]
    lines += [
        f"#SBATCH {flag}={settings[key]}"
        for key, flag in DIRECTIVES
        if settings.get(key)
    ]
    if settings.get("mail_user"):
        lines += [
            "#SBATCH --mail-type=FAIL",
            f"#SBATCH --mail-user={settings['mail_user']}",
        ]

    array = f"0-{n_tasks - 1}"
    if settings.get("max_running"):
        array += f"%{settings['max_running']}"
    lines.append(f"#SBATCH --array={array}")

    # The array index is relative to this chunk, so it is offset back into the
    # full wrapper list, which sed reads 1-based.
    lines += [
        "",
        "export OMP_NUM_THREADS=1",
        "",
        f'WRAPPER=$(sed -n "$((SLURM_ARRAY_TASK_ID + {chunk + 1}))p" {wrapper_list})',
        'echo "Running $WRAPPER"',
        'bash "$WRAPPER"',
        "",
    ]
    return "\n".join(lines)


def submit_slurm(wrapper_scripts, log_dir, settings=None, job_name="ugf-schedule"):
    """Submit the wrappers as one or more Slurm array jobs.

    Parameters
    ----------
    wrapper_scripts : sequence of pathlib.Path
        Scripts to run, one per array task.
    log_dir : str or pathlib.Path
        Directory the wrapper list, batch scripts and logs are written to.
    settings : dict, optional
        The ``[slurm]`` section of the run configuration.
    job_name : str, optional
        Name given to the jobs.
    """
    if not wrapper_scripts:
        log.warning("no wrappers to submit")
        return

    settings = dict(settings or {})
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    wrapper_list = log_dir / "wrapper_list.txt"
    wrapper_list.write_text("".join(f"{path}\n" for path in wrapper_scripts))

    total = len(wrapper_scripts)
    max_array = int(settings.get("max_array", total)) or total
    for chunk in range(0, total, max_array):
        n_tasks = min(max_array, total - chunk)
        script = log_dir / f"slurm_array_{chunk}.sh"
        script.write_text(
            array_script(wrapper_list, log_dir, settings, chunk, n_tasks, job_name)
        )
        try:
            subprocess.run(["sbatch", str(script)], check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            log.error("sbatch failed for %s: %s", script, error)
            return
        log.info("submitted %s with %d tasks", script, n_tasks)
