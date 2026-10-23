"""Checks on the HTCondor submit description generated for m4opt jobs.

The memory and disk requests are ClassAd expressions rather than constants,
so that a job held for exceeding a limit is released and retried with a
larger request. Getting the units or the escalation ceiling wrong fails
silently -- the job is simply never matched -- so both are pinned here, along
with the hold codes the retry is allowed to recover from.
"""

import subprocess
from pathlib import Path

import pytest

from ultrasat_gw_followup.backend import condor
from ultrasat_gw_followup.config import Config

CONFIGS = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def submit_description(monkeypatch):
    """Return the submit file that ``submit_condor_job`` feeds to condor_submit."""
    captured = {}

    def fake_run(args, input=None, **kwargs):
        captured["script"] = input
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def render(**settings):
        condor.submit_condor_job("O5a", 1234, "/logs", "/wrapper.sh", settings)
        return dict(
            line.split(" = ", 1)
            for line in captured["script"].splitlines()
            if " = " in line
        )

    return render


@pytest.mark.parametrize(
    "value,expected_kb",
    [
        ("4096", 4096),
        ("8000 MB", 8192000),
        ("8 GB", 8388608),
        ("80000MB", 81920000),
    ],
)
def test_to_kb(value, expected_kb):
    assert condor.to_kb(value) == expected_kb


def test_to_kb_rejects_garbage():
    with pytest.raises(ValueError):
        condor.to_kb("plenty")


def test_default_requests_escalate_within_a_ceiling(submit_description):
    description = submit_description()
    # request_memory is in MiB, request_disk in KiB: 20000 MB -> 20480000 KiB.
    assert description["request_memory"] == (
        "int(min(200000, 40000 * pow(2.0, NumJobStarts)))"
    )
    assert description["request_disk"] == (
        "int(min(102400000, 20480000 * pow(2.0, NumJobStarts)))"
    )


def test_requests_come_from_the_configuration(submit_description):
    description = submit_description(
        request_memory="80000 MB",
        request_disk="20000 MB",
        max_request_memory="200000 MB",
        max_request_disk="100000 MB",
        growth_factor="1.5",
        max_retries="5",
    )
    assert description["request_memory"] == (
        "int(min(200000, 80000 * pow(1.5, NumJobStarts)))"
    )
    assert description["request_disk"] == (
        "int(min(102400000, 20480000 * pow(1.5, NumJobStarts)))"
    )
    assert "NumJobStarts <= 5" in description["periodic_release"]


def test_ceiling_never_falls_below_the_floor(submit_description):
    """A ceiling below the floor would make the job unmatchable from the start."""
    description = submit_description(
        request_memory="100 GB", max_request_memory="10 GB"
    )
    assert description["request_memory"].startswith("int(min(102400, 102400 *")


def test_held_jobs_are_released_for_memory_disk_and_cplex(submit_description):
    """Exit code 42 reaches the release expression as hold subcode 42."""
    description = submit_description()
    assert description["periodic_release"] == (
        "(HoldReasonCode == 34 || HoldReasonCode == 104 || "
        "(HoldReasonCode == 3 && HoldReasonSubCode == 42)) && (NumJobStarts <= 3)"
    )
    assert description["on_exit_hold_subcode"] == "ExitCode"


def test_accounting_group_and_max_hours_come_from_the_configuration(submit_description):
    description = submit_description(
        accounting_group="ligo.sim.o4.cbc.pe.bayestar", max_hours="12"
    )
    assert description["accounting_group"] == "ligo.sim.o4.cbc.pe.bayestar"
    assert description["+MaxHours"] == "12"


@pytest.mark.parametrize(
    "name,request_memory",
    [
        ("ultrasat_lcs", "80000 MB"),
        ("ultrasat_allsky", "80000 MB"),
        ("uvex", "40000 MB"),
    ],
)
def test_the_run_configs_carry_the_requests_of_the_paper(name, request_memory):
    settings = Config(CONFIGS / f"{name}.ini").getsection("condor")
    assert settings["request_memory"] == request_memory
    assert settings["request_disk"] == "20000 MB"
    assert settings["max_request_memory"] == "200000 MB"
    assert settings["max_request_disk"] == "100000 MB"
    assert settings["max_retries"] == "3"
    assert float(settings["growth_factor"]) == 2.0
