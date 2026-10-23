"""Finding the events to re-run, and turning them back into an event table.

A schedule that is missing, unreadable, or carries the CPLEX 1803 licensing
error is not a result: it has to come back through the scheduler. A plan that
merely stopped on the time limit is a result, and must not.
"""

from pathlib import Path

import pytest
from astropy.table import QTable

from ultrasat_gw_followup.workflow import check_runs, make_redo

RUNS = ("O5a", "O5b")


def write_plan(path, status):
    """Write a one-row schedule carrying a solution status."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plan = QTable({"action": ["observe"]})
    plan.meta["solution_status"] = status
    plan.write(path, overwrite=True)


@pytest.fixture
def run_dir(tmp_path):
    """An output directory holding one schedule per status worth testing."""
    sched = tmp_path / "out" / "schedules"
    write_plan(sched / "O5a" / "1.ecsv", "integer optimal solution")
    write_plan(sched / "O5a" / "2.ecsv", "time limit exceeded")
    write_plan(sched / "O5a" / "3.ecsv", "CPLEX Error  1803: CPLEX Error  1803.")
    (sched / "O5b").mkdir(parents=True, exist_ok=True)
    (sched / "O5b" / "5.ecsv").write_text("not an ecsv file at all")
    # O5b/4.ecsv is deliberately absent.
    return tmp_path / "out"


@pytest.fixture
def events(tmp_path):
    table = QTable(
        {
            "run": ["O5a", "O5a", "O5a", "O5b", "O5b"],
            "coinc_event_id": [1, 2, 3, 4, 5],
            "distance": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )
    path = tmp_path / "data" / "observing-scenarios.ecsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path)
    return path


def test_only_unusable_plans_count_as_crashed(run_dir, events):
    _, crashed = check_runs.check(QTable.read(events), run_dir / "schedules")
    assert crashed == [("O5a", 3), ("O5b", 4), ("O5b", 5)]


def test_a_plan_stopped_on_the_time_limit_is_a_result(run_dir, events):
    statuses, _ = check_runs.check(QTable.read(events), run_dir / "schedules")
    assert statuses["O5a"]["time limit exceeded"] == 1


def test_the_summary_counts_every_event_of_every_run(run_dir, events):
    statuses, _ = check_runs.check(QTable.read(events), run_dir / "schedules")
    assert sorted(statuses) == list(RUNS)
    assert sum(statuses["O5a"].values()) == 3
    assert statuses["O5b"] == {check_runs.MISSING: 1, check_runs.UNREADABLE: 1}


def test_crashed_txt_round_trips(tmp_path):
    crashed = [("O5a", 3), ("O5b", 4)]
    path = check_runs.write_crashed(tmp_path / "crashed.txt", crashed)
    assert path.read_text() == "O5a 3\nO5b 4\n"
    assert check_runs.read_crashed(path) == [("O5a", "3"), ("O5b", "4")]


def test_main_writes_crashed_txt(run_dir, events, capsys):
    status = check_runs.main(
        [
            "--data-dir",
            str(events.parent),
            "--out-dir",
            str(run_dir),
        ]
    )
    assert status == 0
    assert (run_dir / "crashed.txt").read_text() == "O5a 3\nO5b 4\nO5b 5\n"
    assert "3 events to re-run" in capsys.readouterr().out


# --------------------------------------------------------------------------
# make_redo
# --------------------------------------------------------------------------
def test_redo_holds_exactly_the_crashed_events(run_dir, events, capsys):
    check_runs.main(["--data-dir", str(events.parent), "--out-dir", str(run_dir)])
    assert (
        make_redo.main(["--data-dir", str(events.parent), "--out-dir", str(run_dir)])
        == 0
    )

    redo = QTable.read(run_dir / "redo.ecsv")
    assert list(zip(redo["run"], redo["coinc_event_id"])) == [
        ("O5a", 3),
        ("O5b", 4),
        ("O5b", 5),
    ]
    assert redo.colnames == QTable.read(events).colnames
    assert "ugf-scheduler" in capsys.readouterr().out


def test_first_keeps_only_the_head_of_the_list(run_dir, events):
    check_runs.main(["--data-dir", str(events.parent), "--out-dir", str(run_dir)])
    make_redo.main(
        ["--data-dir", str(events.parent), "--out-dir", str(run_dir), "--first", "1"]
    )
    assert len(QTable.read(run_dir / "redo.ecsv")) == 1


def test_make_redo_needs_check_runs_first(tmp_path, events):
    assert (
        make_redo.main(["--data-dir", str(events.parent), "--out-dir", str(tmp_path)])
        == 1
    )


def test_an_unknown_crashed_event_is_an_error(events):
    with pytest.raises(KeyError):
        make_redo.select(QTable.read(events), [("O9z", "1")])


def test_the_redo_table_feeds_the_scheduler_back(run_dir, events):
    """resolve_input must accept the path make_redo prints."""
    from ultrasat_gw_followup.config import resolve_input

    check_runs.main(["--data-dir", str(events.parent), "--out-dir", str(run_dir)])
    make_redo.main(["--data-dir", str(events.parent), "--out-dir", str(run_dir)])
    path = run_dir / "redo.ecsv"
    assert resolve_input(path, Path("elsewhere")) == path
