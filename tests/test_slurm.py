"""The Slurm array scripts: every directive comes from the configuration.

Nothing is hard-coded, so the same code runs on any cluster; what has to be
right is that a directive appears only when the configuration sets it, and
that long runs are split into arrays the scheduler will accept.
"""

from pathlib import Path

import pytest

from ultrasat_gw_followup.backend import slurm
from ultrasat_gw_followup.config import Config

CONFIGS = Path(__file__).resolve().parents[1] / "config"

SETTINGS = {
    "partition": "cpu",
    "account": "bcrv-delta-cpu",
    "mem": "30G",
    "time": "12:00:00",
    "cpus": "1",
    "mail_user": "someone@example.org",
    "max_running": "10",
    "max_array": "1000",
}


@pytest.fixture
def submitted(monkeypatch, tmp_path):
    """Submit wrappers with a fake sbatch; return the scripts it was given."""
    scripts = []
    monkeypatch.setattr(
        slurm.subprocess, "run", lambda args, **kwargs: scripts.append(Path(args[1]))
    )

    def submit(n_wrappers, settings=SETTINGS):
        wrappers = [tmp_path / f"wrapper_{i}.sh" for i in range(n_wrappers)]
        slurm.submit_slurm(wrappers, tmp_path / "logs", settings)
        return [script.read_text() for script in scripts]

    return submit


def directives(script):
    """The ``#SBATCH --flag=value`` lines of a script, as a dict."""
    lines = (line for line in script.splitlines() if line.startswith("#SBATCH "))
    pairs = (line.removeprefix("#SBATCH ").split("=", 1) for line in lines)
    return dict(pair for pair in pairs if len(pair) == 2)


def test_every_directive_comes_from_the_configuration(submitted):
    found = directives(submitted(3)[0])
    assert found["--partition"] == "cpu"
    assert found["--account"] == "bcrv-delta-cpu"
    assert found["--mem"] == "30G"
    assert found["--time"] == "12:00:00"
    assert found["--cpus-per-task"] == "1"
    assert found["--mail-user"] == "someone@example.org"
    assert found["--array"] == "0-2%10"


def test_a_setting_left_blank_writes_no_directive(submitted):
    settings = {key: value for key, value in SETTINGS.items() if key != "account"}
    script = submitted(2, settings)[0]
    assert "--account" not in script
    assert "--partition=cpu" in script


def test_no_mail_user_means_no_mail_directives(submitted):
    settings = {key: value for key, value in SETTINGS.items() if key != "mail_user"}
    script = submitted(2, settings)[0]
    assert "--mail-type" not in script
    assert "--mail-user" not in script


def test_no_max_running_leaves_the_array_unthrottled(submitted):
    settings = {key: value for key, value in SETTINGS.items() if key != "max_running"}
    assert directives(submitted(4, settings)[0])["--array"] == "0-3"


def test_the_wrappers_are_split_into_arrays_of_max_array_tasks(submitted):
    scripts = submitted(7, SETTINGS | {"max_array": "3"})
    assert [directives(script)["--array"] for script in scripts] == [
        "0-2%10",
        "0-2%10",
        "0-0%10",
    ]
    # Each chunk offsets the array index back into the shared wrapper list,
    # which sed reads 1-based.
    assert "SLURM_ARRAY_TASK_ID + 1))p" in scripts[0]
    assert "SLURM_ARRAY_TASK_ID + 4))p" in scripts[1]
    assert "SLURM_ARRAY_TASK_ID + 7))p" in scripts[2]


def test_the_wrapper_list_holds_every_script(submitted, tmp_path):
    submitted(5)
    listed = (tmp_path / "logs" / "wrapper_list.txt").read_text().splitlines()
    assert listed == [str(tmp_path / f"wrapper_{i}.sh") for i in range(5)]


def test_nothing_is_submitted_without_wrappers(submitted):
    assert submitted(0) == []


@pytest.mark.parametrize("name", ["ultrasat_lcs", "ultrasat_allsky", "uvex"])
def test_the_run_configs_carry_a_usable_slurm_section(name):
    settings = Config(CONFIGS / f"{name}.ini").getsection("slurm")
    assert settings["partition"]
    assert int(settings["max_array"]) > 0
    assert int(settings["max_running"]) > 0
