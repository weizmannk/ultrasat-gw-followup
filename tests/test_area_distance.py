"""The percentiles of the text and of the figures must be the same numbers.

The area-distance figures draw a line at each 90th percentile and write it in
a box; ``numbers.tex`` quotes the same five values. They are computed twice,
in two modules, so the convention is pinned here: ``inverted_cdf``, as in the
m4opt paper scripts. Linear interpolation, the default of ``np.percentile``,
shifts the distance percentile of the selected events by about 2% -- enough to
put a figure and its caption at odds.
"""

from pathlib import Path

import numpy as np
import pytest

from ultrasat_gw_followup.analysis import area_distance, mission_yields

LCS = Path(__file__).resolve().parents[1] / "results" / "pixelpop" / "ultrasat_lcs.ecsv"

pytestmark = pytest.mark.skipif(
    not LCS.exists(), reason=f"{LCS} not available in this checkout"
)


@pytest.fixture(scope="module")
def selected():
    """Distance and detection probability of the events selected in O5a."""
    us = mission_yields.load_results(LCS, "ULTRASAT")
    mask = (us.run == "O5a") & us.selected
    return us, us.distance[mask], us.pdet[mask]


def test_the_unweighted_percentile_is_inverted_cdf(selected):
    us, distance, _ = selected
    assert mission_yields.selection_percentiles(us, "O5a")["DistTrig"] == np.quantile(
        distance, 0.9, method="inverted_cdf"
    )


def test_the_weighted_percentile_matches_numpy(selected):
    """``weighted_quantile`` reimplements what the figures call on numpy."""
    us, distance, pdet = selected
    assert mission_yields.selection_percentiles(us, "O5a")["DistDet"] == np.quantile(
        distance, 0.9, weights=pdet, method="inverted_cdf"
    )


def test_no_module_falls_back_on_linear_interpolation():
    """``np.percentile`` interpolates; a copied block of it caused the drift."""
    for module in (area_distance, mission_yields):
        assert "np.percentile" not in Path(module.__file__).read_text()
