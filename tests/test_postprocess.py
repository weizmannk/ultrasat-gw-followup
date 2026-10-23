"""One event, end to end, from an m4opt schedule to the aggregated table.

The fixture is a real ULTRASAT schedule written by m4opt 2.13: 23 rows, 12 of
them observations of 6 distinct sky grid tiles, each visited twice, and a solve
that stopped on "time limit exceeded". It carries the two traits that made
earlier versions of this pipeline fail -- a masked ``field_id`` column, and
masked pointing columns on the slew rows -- so it guards against regressions
that only show up on real data.
"""

import pathlib

import numpy as np
import pytest
from astropy.table import QTable

from ultrasat_gw_followup.probability.detection_probability import observed_fields
from ultrasat_gw_followup.workflow.postprocess import COLUMNS, process

DATA = pathlib.Path(__file__).parent / "data"
SCHED_DIR = DATA / "schedules"
RUN, EVENT_ID = "IR1HLV", 1028

# Positions in the tuple returned by process(), which main() unpacks into
# one output column each -- so a shift here is a silently mislabelled table.
DETECTION_PROBABILITY, SOLUTION_STATUS = 0, 3
NUM_FIELDS, FIELD_IDS = 5, 6
MISSION, VISITS, SKYGRID = 9, 18, 20
EXPECTED_LENGTH = len(COLUMNS)


@pytest.fixture
def plan():
    return QTable.read(SCHED_DIR / RUN / f"{EVENT_ID}.ecsv")


@pytest.fixture
def event_row(plan):
    """An event located at the centre of one of the observed fields.

    Pointing the event at a field the schedule actually covers is what makes
    the detection probability exercise the limiting-magnitude path rather than
    returning early.
    """
    target = observed_fields(plan)["target_coord"][0]
    table = QTable(
        {
            "run": [RUN],
            "coinc_event_id": [EVENT_ID],
            "longitude": [target.ra.rad],
            "latitude": [target.dec.rad],
            "distance": [100.0],
        }
    )
    return table[0]


def test_the_fixture_is_the_schedule_we_think_it_is(plan):
    assert len(plan) == 23
    assert plan.meta["solution_status"] == "time limit exceeded"
    # The slew rows name no field; that mask is what used to break .filled().
    slews = plan[plan["action"] == "slew"]
    assert np.all(np.ma.getmaskarray(slews["field_id"]))


def test_process_returns_one_value_per_output_column(event_row):
    assert len(process(event_row, SCHED_DIR)) == EXPECTED_LENGTH


def test_process_counts_the_fields_that_were_observed(event_row):
    result = process(event_row, SCHED_DIR)
    assert result[NUM_FIELDS] == 6
    np.testing.assert_array_equal(result[FIELD_IDS], [12, 26, 28, 44, 67, 90])


def test_field_ids_index_the_mission_sky_grid(event_row):
    """The identifier is the mission's own name for the tile, not a local one."""
    missions = pytest.importorskip("m4opt.missions")
    result = process(event_row, SCHED_DIR)
    grid = missions.ultrasat.skygrid["non-overlap"]
    assert result[FIELD_IDS].max() < len(grid)


def test_process_carries_the_solver_metadata_through(event_row):
    result = process(event_row, SCHED_DIR)
    assert result[SOLUTION_STATUS] == "time limit exceeded"
    assert result[MISSION] == "ultrasat"
    assert result[SKYGRID] == "non-overlap"
    assert result[VISITS] == 2


def test_an_event_inside_an_observed_field_gets_a_real_probability(event_row):
    probability = process(event_row, SCHED_DIR)[DETECTION_PROBABILITY]
    assert 0.0 < probability <= 1.0


def test_a_missing_schedule_yields_nulls_and_no_fields(event_row):
    event_row["coinc_event_id"] = 999999
    result = process(event_row, SCHED_DIR)

    assert len(result) == EXPECTED_LENGTH
    assert result[DETECTION_PROBABILITY] is None
    assert result[NUM_FIELDS] is None
    # Not None: a None cannot be written to the variable-length ECSV column.
    assert len(result[FIELD_IDS]) == 0


def test_the_aggregated_table_survives_a_round_trip(event_row, tmp_path):
    """field_ids has to reach the ECSV that analyses are run from."""
    found = process(event_row, SCHED_DIR)
    event_row["coinc_event_id"] = 999999
    missing = process(event_row, SCHED_DIR)

    table = QTable(
        {
            "coinc_event_id": [EVENT_ID, 999999],
            "num_fields": [found[NUM_FIELDS], missing[NUM_FIELDS]],
            "field_ids": [found[FIELD_IDS], missing[FIELD_IDS]],
        }
    )
    path = tmp_path / "aggregated.ecsv"
    table.write(path)

    back = QTable.read(path)
    np.testing.assert_array_equal(back["field_ids"][0], [12, 26, 28, 44, 67, 90])
    assert len(back["field_ids"][1]) == 0


def test_the_column_names_are_the_ones_the_analyses_read():
    for name in (
        "detection_probability_known_position",
        "objective_value",
        "best_bound",
        "solution_status",
        "num_fields",
        "cutoff",
        "skygrid",
    ):
        assert name in COLUMNS


def test_the_network_snr_survives_the_scheduler_threshold(event_row, tmp_path):
    """postprocess overwrites "snr" with the kilonova threshold, so it keeps a copy."""
    from ultrasat_gw_followup.workflow import postprocess

    events = QTable(
        {
            "run": [RUN],
            "coinc_event_id": [EVENT_ID],
            "longitude": [event_row["longitude"]],
            "latitude": [event_row["latitude"]],
            "distance": [event_row["distance"]],
            "snr": [11.5],
        }
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    events.write(data_dir / "observing-scenarios.ecsv")

    assert (
        postprocess.main(
            [
                "--data-dir",
                str(data_dir),
                "--out-dir",
                str(DATA),
                "--sched-dir",
                "schedules",
                "--output-file",
                str(tmp_path / "result.ecsv"),
            ]
        )
        == 0
    )

    result = QTable.read(tmp_path / "result.ecsv")
    assert result["network_snr"][0] == 11.5
    assert result["snr"][0] != 11.5
