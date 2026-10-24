"""The scheduling command and the wrapper script built around it.

What the paper depends on is that the three run configurations produce
exactly the m4opt command they describe, and that the wrapper turns a CPLEX
1803 licensing failure -- which m4opt reports with exit status 0 -- into a
failed job.
"""

import shlex
import subprocess
from pathlib import Path

import pytest
from astropy.table import QTable

from ultrasat_gw_followup.workflow import scheduler

CONFIGS = Path(__file__).resolve().parents[1] / "config"

#: Settings the three runs of the paper share.
COMMON = {
    "--bandpass": "NUV",
    "--nside": "128",
    "--absmag-stdev": "1.3",
    "--snr": "5.0",
    "--delay": "15min",
    "--deadline": "6hour",
    "--exptime-min": "300 s",
    "--exptime-max": "inf s",
    "--cutoff": "0.1",
    "--visits": "2",
    "--cadence": "30min",
    "--timelimit": "6hour",
    "--memory": "30GiB",
    "--jobs": "0",
}


def options(command):
    """The ``--flag=value`` tokens of a command, as a dict."""
    pairs = (token.split("=", 1) for token in command if token.startswith("--"))
    return dict(pair for pair in pairs if len(pair) == 2)


def command_for(name, tmp_path, extra=()):
    """Build the m4opt command of one run configuration."""
    args = scheduler.parse_arguments(["--config", str(CONFIGS / f"{name}.ini"), *extra])
    return scheduler.build_command(
        "m4opt",
        tmp_path / "1234.fits",
        tmp_path / "1234.ecsv",
        tmp_path / "PROGRESS_1234.ecsv",
        args,
    )


def arguments_for(name, tmp_path, extra=()):
    """Parse one run configuration, redirecting its directories to ``tmp_path``."""
    return scheduler.parse_arguments(
        [
            "--config",
            str(CONFIGS / f"{name}.ini"),
            "--data-dir",
            str(tmp_path / "data"),
            "--out-dir",
            str(tmp_path / "out"),
            *extra,
        ]
    )


@pytest.mark.parametrize("name", ["ultrasat_lcs", "ultrasat_allsky", "uvex"])
def test_the_runs_share_the_settings_the_comparison_controls(name, tmp_path):
    command = command_for(name, tmp_path)
    found = options(command)
    for flag, value in COMMON.items():
        assert found[flag] == value, flag
    assert "--appmag-dist" in command


@pytest.mark.parametrize(
    "name,mission,skygrid,absmag",
    [
        ("ultrasat_lcs", "ultrasat", "non-overlap", "-16.0"),
        ("ultrasat_allsky", "ultrasat", "allsky", "-16.0"),
        ("uvex", "uvex", None, "-15.6"),
    ],
)
def test_each_run_differs_only_in_mission_grid_and_magnitude(
    name, mission, skygrid, absmag, tmp_path
):
    found = options(command_for(name, tmp_path))
    assert found["--mission"] == mission
    assert found.get("--skygrid") == skygrid
    assert found["--absmag-mean"] == absmag


def test_no_max_fields_flag_when_the_config_leaves_it_blank(tmp_path):
    assert "--max-fields" not in options(command_for("uvex", tmp_path))


def test_the_command_line_overrides_the_config(tmp_path):
    command = command_for("uvex", tmp_path, ["--snr", "8", "--no-appmag-dist"])
    assert options(command)["--snr"] == "8.0"
    assert "--no-appmag-dist" in command


def test_the_sky_map_and_the_outputs_are_positional_or_explicit(tmp_path):
    command = command_for("uvex", tmp_path)
    assert command[:2] == ["m4opt", "schedule"]
    assert command[2:4] == [str(tmp_path / "1234.fits"), str(tmp_path / "1234.ecsv")]
    assert options(command)["--write-progress"] == str(tmp_path / "PROGRESS_1234.ecsv")


# --------------------------------------------------------------------------
# The wrapper
# --------------------------------------------------------------------------
def test_the_wrapper_sets_tmpdir_to_the_condor_scratch_directory():
    text = scheduler.wrapper_text(["m4opt", "schedule"], "plan.ecsv")
    assert 'export TMPDIR="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"' in text


def test_the_wrapper_quotes_the_command(tmp_path):
    command = ["m4opt", "schedule", str(tmp_path / "a b.fits"), "--exptime-min=300 s"]
    text = scheduler.wrapper_text(command, tmp_path / "plan.ecsv")
    assert shlex.join(command) in text


def run_wrapper(tmp_path, plan_text, status=0):
    """Run a wrapper whose m4opt call is a stub, and return its exit status."""
    plan = tmp_path / "plan.ecsv"
    plan.write_text(plan_text)
    stub = tmp_path / "fake-m4opt"
    stub.write_text(f"#!/bin/bash\nexit {status}\n")
    stub.chmod(0o755)

    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(scheduler.wrapper_text([str(stub), "schedule"], plan))
    return subprocess.run(["bash", str(wrapper)], check=False).returncode


def test_a_plan_carrying_cplex_error_1803_fails_the_job(tmp_path):
    plan = "# - {solution_status: 'CPLEX Error  1803: CPLEX Error  1803.'}\n"
    assert run_wrapper(tmp_path, plan) == scheduler.CPLEX_ERROR_EXIT_CODE


def test_a_healthy_plan_succeeds(tmp_path):
    assert run_wrapper(tmp_path, "# - {solution_status: optimal}\n") == 0


def test_a_failing_m4opt_keeps_its_own_status(tmp_path):
    assert run_wrapper(tmp_path, "# - {solution_status: optimal}\n", status=3) == 3


# --------------------------------------------------------------------------
# Event selection and dry run
# --------------------------------------------------------------------------
def test_skip_existing_leaves_the_events_already_scheduled_alone(tmp_path):
    args = arguments_for("uvex", tmp_path, ["--skip-existing"])
    paths = scheduler.make_directories(args)
    for event_id in (1, 2):
        (paths["skymap"] / "O5a").mkdir(parents=True, exist_ok=True)
        (paths["skymap"] / "O5a" / f"{event_id}.fits").touch()
    (paths["sched"] / "O5a").mkdir(parents=True, exist_ok=True)
    (paths["sched"] / "O5a" / "1.ecsv").touch()

    table = QTable({"run": ["O5a", "O5a"], "coinc_event_id": [1, 2]})
    assert scheduler.plan_events(table, paths, args) == [("O5a", 2)]
    args.skip_existing = False
    assert scheduler.plan_events(table, paths, args) == [("O5a", 1), ("O5a", 2)]


def test_an_event_without_a_sky_map_is_skipped(tmp_path):
    args = arguments_for("uvex", tmp_path)
    paths = scheduler.make_directories(args)
    table = QTable({"run": ["O5a"], "coinc_event_id": [1]})
    assert scheduler.plan_events(table, paths, args) == []


def test_dry_run_writes_the_wrappers_and_submits_nothing(tmp_path, monkeypatch):
    args = arguments_for("uvex", tmp_path, ["--dry-run"])
    paths = scheduler.make_directories(args)
    (paths["skymap"] / "O5a").mkdir(parents=True, exist_ok=True)
    (paths["skymap"] / "O5a" / "1234.fits").touch()
    event_table = args.data_dir / "observing-scenarios.ecsv"
    QTable({"run": ["O5a"], "coinc_event_id": [1234]}).write(event_table)

    monkeypatch.setattr(scheduler, "find_m4opt", lambda: "m4opt")
    monkeypatch.setattr(
        scheduler, "submit", lambda *a: pytest.fail("dry run must not submit")
    )
    assert (
        scheduler.main(
            [
                "--config",
                str(CONFIGS / "uvex.ini"),
                "--data-dir",
                str(args.data_dir),
                "--out-dir",
                str(args.out_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    wrapper = paths["log"] / "wrapper_O5a_1234.sh"
    assert wrapper.is_file()
    text = wrapper.read_text()
    assert "--mission=uvex" in text
    assert str(paths["sched"] / "O5a" / "1234.ecsv") in text


# --------------------------------------------------------------------------
# CPLEX preflight
# --------------------------------------------------------------------------
def prepare_run(tmp_path, extra=()):
    """A one-event run, ready to submit."""
    args = arguments_for("uvex", tmp_path, extra)
    paths = scheduler.make_directories(args)
    (paths["skymap"] / "O5a").mkdir(parents=True, exist_ok=True)
    (paths["skymap"] / "O5a" / "1234.fits").touch()
    QTable({"run": ["O5a"], "coinc_event_id": [1234]}).write(
        args.data_dir / "observing-scenarios.ecsv"
    )
    return [
        "--config",
        str(CONFIGS / "uvex.ini"),
        "--data-dir",
        str(args.data_dir),
        "--out-dir",
        str(args.out_dir),
        *extra,
    ]


def crippled_cplex():
    raise RuntimeError("CPLEX cannot solve a 3000-variable model")


def test_a_crippled_cplex_stops_the_submission(tmp_path, monkeypatch):
    """1975 jobs dying one by one on the same error is the alternative."""
    argv = prepare_run(tmp_path)
    monkeypatch.setattr(scheduler, "find_m4opt", lambda: "m4opt")
    monkeypatch.setattr(scheduler, "check_cplex", crippled_cplex)
    monkeypatch.setattr(
        scheduler, "submit", lambda *a: pytest.fail("must not submit without CPLEX")
    )
    assert scheduler.main(argv) == 1


def test_skip_cplex_check_submits_anyway(tmp_path, monkeypatch):
    argv = prepare_run(tmp_path, ["--skip-cplex-check"])
    submitted = []
    monkeypatch.setattr(scheduler, "find_m4opt", lambda: "m4opt")
    monkeypatch.setattr(scheduler, "check_cplex", crippled_cplex)
    monkeypatch.setattr(scheduler, "submit", lambda *a: submitted.append(a))
    assert scheduler.main(argv) == 0
    assert submitted


def test_a_dry_run_does_not_need_cplex(tmp_path, monkeypatch):
    """Writing wrappers is useful on a machine with no solver at all."""
    argv = prepare_run(tmp_path, ["--dry-run"])
    monkeypatch.setattr(scheduler, "find_m4opt", lambda: "m4opt")
    monkeypatch.setattr(scheduler, "check_cplex", crippled_cplex)
    assert scheduler.main(argv) == 0


def test_the_probe_is_over_the_crippled_limit():
    """Below 1000 variables the size-limited edition would pass the check."""
    assert scheduler.CPLEX_PROBE_SIZE > 1000
