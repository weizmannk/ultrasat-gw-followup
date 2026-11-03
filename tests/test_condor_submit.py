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
        "ifthenelse(MemoryUsage =!= undefined, "
        "int(min({200000, max({40000, MemoryUsage * 2.0})})), 40000)"
    )
    assert description["request_disk"] == (
        "ifthenelse(DiskUsage =!= undefined, "
        "int(min({102400000, max({20480000, DiskUsage * 2.0})})), 20480000)"
    )


@pytest.mark.parametrize(
    "attribute,usage,floor",
    [("request_memory", "MemoryUsage", 40000), ("request_disk", "DiskUsage", 20480000)],
)
def test_the_request_is_a_plain_number_before_the_first_attempt(
    submit_description, attribute, usage, floor
):
    """The measured usage is undefined until the job has run.

    Without the guard the expression evaluates to undefined and the schedd
    refuses the job: "Submit requirement REQUESTDISK evaluated to
    non-boolean".
    """
    expression = submit_description()[attribute]
    assert expression.startswith(f"ifthenelse({usage} =!= undefined, ")
    assert expression.endswith(f"), {floor})")


@pytest.mark.parametrize(
    "attribute,usage",
    [("request_memory", "MemoryUsage"), ("request_disk", "DiskUsage")],
)
def test_a_retry_asks_for_a_multiple_of_what_was_measured(
    submit_description, attribute, usage
):
    """Escalating on the attempt number would ignore what the job needed."""
    assert f"{usage} * 2.0" in submit_description()[attribute]


@pytest.mark.parametrize("attribute", ["request_memory", "request_disk"])
def test_min_and_max_are_given_lists(submit_description, attribute):
    """ClassAd min() takes a list; min(a, b) is an error, not a minimum."""
    expression = submit_description()[attribute]
    assert "min({" in expression
    assert "max({" in expression


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
        "ifthenelse(MemoryUsage =!= undefined, "
        "int(min({200000, max({80000, MemoryUsage * 1.5})})), 80000)"
    )
    assert description["request_disk"] == (
        "ifthenelse(DiskUsage =!= undefined, "
        "int(min({102400000, max({20480000, DiskUsage * 1.5})})), 20480000)"
    )
    assert "NumJobStarts <= 5" in description["periodic_release"]


def test_ceiling_never_falls_below_the_floor(submit_description):
    """A ceiling below the floor would make the job unmatchable from the start."""
    description = submit_description(
        request_memory="100 GB", max_request_memory="10 GB"
    )
    assert "min({102400, max({102400," in description["request_memory"]


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


RUNS = ("ultrasat_lcs", "ultrasat_allsky", "uvex")

# The floors and ceilings are tuned against the pool as runs come back, so no
# absolute value is pinned here: a test that must be edited every time one is
# raised only teaches people to edit tests. What is pinned is what breaks a
# run, and what has to stay consistent between the three.


def condor_settings(name):
    return Config(CONFIGS / f"{name}.ini").getsection("condor")


@pytest.mark.parametrize("name", RUNS)
def test_the_retry_policy_is_the_one_of_the_paper(name):
    settings = condor_settings(name)
    assert settings["max_retries"] == "3"
    assert float(settings["growth_factor"]) == 2.0


def test_the_ceilings_are_the_same_for_every_run():
    """The pool is one, so only the floors may differ between runs."""
    ceilings = {
        name: (
            condor_settings(name)["max_request_memory"],
            condor_settings(name)["max_request_disk"],
        )
        for name in RUNS
    }
    assert len(set(ceilings.values())) == 1, ceilings


@pytest.mark.parametrize("name", RUNS)
def test_no_floor_exceeds_its_ceiling(name):
    """A floor above the ceiling would make the job unmatchable from the start."""
    settings = condor_settings(name)
    for request, ceiling in (
        ("request_memory", "max_request_memory"),
        ("request_disk", "max_request_disk"),
    ):
        floor, cap = condor.to_kb(settings[request]), condor.to_kb(settings[ceiling])
        assert 0 < floor <= cap, f"{name}: {request} above its ceiling"


@pytest.mark.parametrize("name", RUNS)
def test_a_floor_leaves_room_for_at_least_one_escalation(name):
    """A floor above cap/growth spends all its retries re-requesting the cap."""
    settings = condor_settings(name)
    growth = float(settings["growth_factor"])
    for request, ceiling in (
        ("request_memory", "max_request_memory"),
        ("request_disk", "max_request_disk"),
    ):
        floor, cap = condor.to_kb(settings[request]), condor.to_kb(settings[ceiling])
        assert floor * growth <= cap, f"{name}: {request} cannot double"
