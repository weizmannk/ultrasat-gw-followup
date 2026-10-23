"""The two known-position detection metrics, on a real two-visit schedule.

``get_detection_probability_known_position`` asks whether the kilonova was
above threshold in the deepest field of the plan.
``get_detection_probability_two_visits`` asks the stricter question that
identifying a transient actually requires: was it above threshold in *every*
visit of some field, so that it can be told apart from a moving Solar System
object.

The fixture is the ULTRASAT schedule of ``tests/test_postprocess.py``: six sky
grid tiles, each visited twice, with unequal durations between visits.
"""

import pathlib

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Distance
from astropy.table import QTable, vstack
from scipy import stats

from ultrasat_gw_followup.probability.detection_probability import (
    _limmag_per_row,
    get_detection_probability_known_position,
    get_detection_probability_two_visits,
    observed_fields,
)

DATA = pathlib.Path(__file__).parent / "data"
SCHED = DATA / "schedules" / "IR1HLV" / "1028.ecsv"
DISTANCE_MPC = 100.0


@pytest.fixture
def plan():
    return QTable.read(SCHED)


@pytest.fixture
def plan_args(plan):
    args = dict(plan.meta["args"])
    args.pop("skymap", None)
    return args


def event_at(coord, distance=DISTANCE_MPC):
    """A one-row event table centred on ``coord``."""
    return QTable(
        {
            "longitude": [coord.ra.rad],
            "latitude": [coord.dec.rad],
            "distance": [distance],
        }
    )[0]


def pointing_groups(observations):
    """Row indices of each distinct pointing, in table order."""
    coords = observations["target_coord"]
    _, inverse = np.unique(
        np.stack([coords.ra.deg, coords.dec.deg], axis=-1),
        axis=0,
        return_inverse=True,
    )
    inverse = np.ravel(inverse)
    return [np.flatnonzero(inverse == g) for g in np.unique(inverse)]


def probability_at(limmag, plan_args, distance=DISTANCE_MPC):
    """The CDF the metrics are expected to return for a given limit."""
    lim_absmag = limmag - Distance(distance * u.Mpc).distmod
    return stats.norm(
        loc=plan_args["absmag_mean"], scale=plan_args["absmag_stdev"]
    ).cdf(lim_absmag.to_value(u.mag))


# --------------------------------------------------------------------------
# The fixture is what the tests assume
# --------------------------------------------------------------------------
def test_fixture_has_two_visits_per_field(plan):
    observations = observed_fields(plan)
    groups = pointing_groups(observations)
    assert len(groups) == 6
    assert all(len(group) == 2 for group in groups)
    assert plan.meta["args"]["visits"] == 2


# --------------------------------------------------------------------------
# Single visit: the two metrics are the same question
# --------------------------------------------------------------------------
def test_metrics_agree_when_every_field_is_observed_once(plan, plan_args):
    observations = observed_fields(plan)
    first_visits = observations[[group[0] for group in pointing_groups(observations)]]
    event = event_at(first_visits["target_coord"][0])

    known = get_detection_probability_known_position(first_visits, event, plan_args)
    two = get_detection_probability_two_visits(first_visits, event, plan_args)
    assert two == known


# --------------------------------------------------------------------------
# Two visits: the stricter metric is set by the shallower one
# --------------------------------------------------------------------------
@pytest.fixture
def one_field_two_visits(plan):
    """One pointing observed twice, the longer visit first.

    Ordering matters: ``astropy.table.unique`` keeps the first row of each
    group, so the existing metric reads the *first* visit rather than the
    longest. Putting the longer one first is what makes the two metrics
    disagree, and is the case worth testing.
    """
    observations = observed_fields(plan)
    group = pointing_groups(observations)[0]
    pair = observations[list(group)]
    order = np.argsort(-pair["duration"].to_value(u.s))
    return pair[list(order)]


def test_two_visit_metric_uses_the_shorter_visit(one_field_two_visits, plan_args):
    plan = one_field_two_visits
    event = event_at(plan["target_coord"][0])
    limmags = _limmag_per_row(plan, _mission(plan_args), plan_args)
    assert limmags[0] > limmags[1], "the longer visit must be the deeper one"

    two = get_detection_probability_two_visits(plan, event, plan_args)
    assert two == pytest.approx(probability_at(limmags.min(), plan_args))


def test_known_position_metric_uses_the_first_visit(one_field_two_visits, plan_args):
    plan = one_field_two_visits
    event = event_at(plan["target_coord"][0])
    limmags = _limmag_per_row(plan, _mission(plan_args), plan_args)

    known = get_detection_probability_known_position(plan, event, plan_args)
    assert known == pytest.approx(probability_at(limmags[0], plan_args))


def test_the_two_metrics_actually_differ_here(one_field_two_visits, plan_args):
    plan = one_field_two_visits
    event = event_at(plan["target_coord"][0])
    known = get_detection_probability_known_position(plan, event, plan_args)
    two = get_detection_probability_two_visits(plan, event, plan_args)
    assert two < known


# --------------------------------------------------------------------------
# Ordering property, on the full schedule
# --------------------------------------------------------------------------
@pytest.mark.parametrize("i_field", range(6))
def test_two_visits_never_exceeds_known_position(plan, plan_args, i_field):
    observations = observed_fields(plan)
    groups = pointing_groups(observations)
    event = event_at(observations["target_coord"][groups[i_field][0]])

    known = get_detection_probability_known_position(plan, event, plan_args)
    two = get_detection_probability_two_visits(plan, event, plan_args)
    assert two <= known


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------
def test_empty_plan_returns_zero(plan, plan_args):
    empty = plan[:0]
    event = event_at(observed_fields(plan)["target_coord"][0])
    assert get_detection_probability_known_position(empty, event, plan_args) == 0
    assert get_detection_probability_two_visits(empty, event, plan_args) == 0


def test_uncovered_position_returns_zero(plan, plan_args):
    """An event on the opposite side of the sky from every pointing."""
    observed = observed_fields(plan)["target_coord"][0]
    antipode = QTable(
        {
            "longitude": [(observed.ra + 180 * u.deg).wrap_at(360 * u.deg).rad],
            "latitude": [-observed.dec.rad],
            "distance": [DISTANCE_MPC],
        }
    )[0]
    assert get_detection_probability_known_position(plan, antipode, plan_args) == 0
    assert get_detection_probability_two_visits(plan, antipode, plan_args) == 0


def test_duplicated_row_does_not_change_the_two_visit_result(plan_args, plan):
    """Adding an identical third visit changes nothing: the min is unchanged."""
    observations = observed_fields(plan)
    group = pointing_groups(observations)[0]
    pair = observations[list(group)]
    event = event_at(pair["target_coord"][0])

    before = get_detection_probability_two_visits(pair, event, plan_args)
    tripled = vstack([pair, pair[-1:]])
    after = get_detection_probability_two_visits(tripled, event, plan_args)
    assert after == pytest.approx(before)


def _mission(plan_args):
    from m4opt import missions

    return getattr(missions, plan_args["mission"])


# --------------------------------------------------------------------------
# Grouping key
# --------------------------------------------------------------------------
def test_field_id_and_coordinates_give_the_same_visit_grouping(plan):
    """The implementation groups by field_id; the tests group by coordinates.

    They must agree, otherwise one of the two is wrong. Checking it here is
    what makes the coordinate-based helper above a real independent check
    rather than a copy of the implementation.
    """
    from ultrasat_gw_followup.probability.detection_probability import _visit_groups

    observations = observed_fields(plan)
    by_field_id = _visit_groups(observations)
    assert "field_id" in observations.colnames, "fixture should exercise field_id"

    for group in pointing_groups(observations):
        assert len(set(by_field_id[group])) == 1, group
    for key in np.unique(by_field_id):
        rows = np.flatnonzero(by_field_id == key)
        assert any(
            np.array_equal(rows, group) for group in pointing_groups(observations)
        )


def test_grouping_falls_back_to_coordinates_without_field_id(plan, plan_args):
    """Schedules written before m4opt 2.13 have no field_id column."""
    observations = observed_fields(plan)
    without = observations.copy()
    without.remove_column("field_id")
    event = event_at(without["target_coord"][0])

    assert get_detection_probability_two_visits(
        without, event, plan_args
    ) == pytest.approx(get_detection_probability_two_visits(plan, event, plan_args))
