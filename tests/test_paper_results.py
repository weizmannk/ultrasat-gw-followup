"""The merger rate must follow the population model of the sample.

The GWTC-5.0 FullPop and PixelPop releases have different median rates, and
every lambda in the paper scales with that number. The model is recorded in
the event table by the unpacker, so a table can never be weighted with the
wrong rate.
"""

from pathlib import Path

import pytest
from astropy.table import QTable

from ultrasat_gw_followup.probability.rate import MERGER_RATE
from ultrasat_gw_followup.workflow import paper_results

RESULTS = Path(__file__).resolve().parents[1] / "results"
LCS = RESULTS / "ultrasat_lcs.ecsv"

pytestmark = pytest.mark.skipif(
    not LCS.exists(), reason=f"{LCS} not available in this checkout"
)


def test_the_priors_are_the_published_quantiles():
    """Medians of the GWTC-5.0 rate summary, not the FullPop MAP (65.09)."""
    assert MERGER_RATE == {
        "fullpop": {"lo": 50.0, "mid": 100.0, "hi": 203.0},
        "pixelpop": {"lo": 39.0, "mid": 68.0, "hi": 139.0},
    }


def test_paper_results_does_not_restate_the_rates():
    """A second copy of the median is what let a wrong PixelPop value through."""
    assert "MERGER_RATE = {" not in Path(paper_results.__file__).read_text()


def test_a_table_without_a_model_is_read_as_fullpop():
    """The tables shipped in results/ predate the metadata; their numbers stand."""
    assert "population_model" not in QTable.read(LCS).meta
    assert paper_results.load_results(LCS, "ULTRASAT").population == "fullpop"


def test_the_rate_follows_the_recorded_model(tmp_path):
    table = QTable.read(LCS)
    reference = paper_results.load_results(LCS, "ULTRASAT")

    table.meta["population_model"] = "pixelpop"
    path = tmp_path / "pixelpop.ecsv"
    table.write(path)
    pixelpop = paper_results.load_results(path, "ULTRASAT")

    assert pixelpop.population == "pixelpop"
    assert pixelpop.n_detected() == pytest.approx(0.68 * reference.n_detected())


def test_an_unknown_model_is_refused(tmp_path):
    table = QTable.read(LCS)
    table.meta["population_model"] = "handwaving"
    path = tmp_path / "bad.ecsv"
    table.write(path)
    with pytest.raises(SystemExit, match="unknown population model"):
        paper_results.load_results(path, "ULTRASAT")


def test_missions_from_different_models_cannot_be_compared(tmp_path):
    table = QTable.read(LCS)
    table.meta["population_model"] = "pixelpop"
    path = tmp_path / "pixelpop.ecsv"
    table.write(path)

    with pytest.raises(SystemExit, match="merger-rate priors differ"):
        paper_results.check_consistency(
            paper_results.load_results(LCS, "ULTRASAT"),
            paper_results.load_results(path, "UVEX"),
        )
