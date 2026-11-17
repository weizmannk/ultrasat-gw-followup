"""The grid of candidate pointings covers the sky at any roll angle.

The circle inscribed in the square field stays inside the field however the
spacecraft is rolled, so a grid covers the sky at every orientation exactly
when no position is further from the nearest centre than that radius. The
paper quotes the two numbers this gives, 362 pointings and 6.83 degrees, so
they are pinned here.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import RectangleSkyRegion

import sky_grid

#: Radius of the circle inscribed in the 14.28 degree ULTRASAT field.
RADIUS = 7.14 * u.deg


@pytest.fixture(scope="module")
def mission():
    return pytest.importorskip("m4opt.missions").ultrasat


@pytest.fixture(scope="module")
def geodesic():
    return sky_grid.smallest_covering_grid(RADIUS)


def test_covering_radius_of_an_octahedron():
    """Exact case: the cube corners are farthest from the six axis directions."""
    centers = SkyCoord([0, 90, 180, 270, 0, 0] * u.deg, [0, 0, 0, 0, 90, -90] * u.deg)
    expected = np.degrees(np.arccos(1 / np.sqrt(3)))
    assert sky_grid.covering_radius(centers).to_value(u.deg) == pytest.approx(expected)


def test_the_inscribed_radius_is_half_the_field(mission):
    assert mission.fov.width.to_value(u.deg) == pytest.approx(14.28)
    assert sky_grid.inscribed_radius(mission.fov).to_value(u.deg) == pytest.approx(7.14)


def test_the_hexagon_bound_is_where_the_search_starts():
    """M4OPT rounds this to the 132 deg2 it asks geodesic() for."""
    assert sky_grid.hexagon_area(RADIUS).to_value(u.deg**2) == pytest.approx(
        132.45, abs=0.01
    )


def test_the_grid_is_the_one_the_paper_quotes(geodesic):
    assert len(geodesic) == 362
    radius = sky_grid.covering_radius(geodesic)
    assert radius <= RADIUS
    assert radius.to_value(u.deg) == pytest.approx(6.83, abs=0.005)


def test_no_smaller_geodesic_grid_covers_the_sky():
    """The grid below it in the same family leaves part of the sky bare."""
    skygrid = pytest.importorskip("m4opt").skygrid
    smaller = skygrid.geodesic(sky_grid.hexagon_area(RADIUS), class_="III")
    assert len(smaller) < 362
    assert sky_grid.covering_radius(smaller) > RADIUS


def test_the_grid_leaves_nothing_uncovered(geodesic):
    counts, _ = sky_grid.coverage_map(geodesic, RADIUS, nside=64)
    assert counts.min() >= 1


def test_it_agrees_with_the_mission_once_m4opt_defines_it(mission, geodesic):
    """The runs read their grid from the mission, so the two must match.

    M4OPT defines ``geodesic`` for ULTRASAT only after 4.0.1; until the pin
    moves there is nothing to compare with.
    """
    centers = mission.skygrid.get("geodesic")
    if centers is None:
        pytest.skip(f"no 'geodesic' grid in this m4opt: {list(mission.skygrid)}")
    assert len(centers) == len(geodesic)
    assert np.allclose(sky_grid.unit_vectors(centers), sky_grid.unit_vectors(geodesic))


def test_the_survey_grids_are_a_different_trade(mission):
    """AllSS covers at any roll; the non-overlapping LCS tiling does not."""
    radius = sky_grid.inscribed_radius(mission.fov)
    assert sky_grid.covering_radius(mission.skygrid["allsky"]) <= radius
    assert sky_grid.covering_radius(mission.skygrid["non-overlap"]) > radius


def test_outline_traces_the_edges_of_the_field():
    fov = RectangleSkyRegion(
        SkyCoord(0 * u.deg, 0 * u.deg), width=14.28 * u.deg, height=14.28 * u.deg
    )
    vertices = sky_grid.outline(fov).vertices
    assert np.abs(vertices.ra.wrap_at(180 * u.deg).deg).max() == pytest.approx(7.14)
    assert np.abs(vertices.dec.deg).max() == pytest.approx(7.14)
