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
PIXELPOP = RESULTS / "pixelpop" / "ultrasat_lcs.ecsv"
FULLPOP = RESULTS / "fullpop" / "ultrasat_lcs.ecsv"
LCS = PIXELPOP

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


def test_the_pixelpop_tables_declare_their_model():
    """postprocess carries the model the unpacker recorded all the way through."""
    assert QTable.read(PIXELPOP).meta["population_model"] == "pixelpop"
    assert paper_results.load_results(PIXELPOP, "ULTRASAT").population == "pixelpop"


@pytest.mark.skipif(not FULLPOP.exists(), reason="no fullpop tables in this checkout")
def test_a_table_without_a_model_is_read_as_fullpop():
    """The fullpop tables predate the metadata; their numbers still stand."""
    assert "population_model" not in QTable.read(FULLPOP).meta
    assert paper_results.load_results(FULLPOP, "ULTRASAT").population == "fullpop"


def test_the_rate_follows_the_recorded_model(tmp_path):
    """Relabelling a table rescales every count by the ratio of the medians."""
    reference = paper_results.load_results(LCS, "ULTRASAT")

    table = QTable.read(LCS)
    table.meta["population_model"] = "fullpop"
    path = tmp_path / "fullpop.ecsv"
    table.write(path)
    relabelled = paper_results.load_results(path, "ULTRASAT")

    assert relabelled.population == "fullpop"
    ratio = MERGER_RATE["fullpop"]["mid"] / MERGER_RATE["pixelpop"]["mid"]
    assert relabelled.n_detected() == pytest.approx(ratio * reference.n_detected())


def test_an_unknown_model_is_refused(tmp_path):
    table = QTable.read(LCS)
    table.meta["population_model"] = "handwaving"
    path = tmp_path / "bad.ecsv"
    table.write(path)
    with pytest.raises(SystemExit, match="unknown population model"):
        paper_results.load_results(path, "ULTRASAT")


def test_missions_from_different_models_cannot_be_compared(tmp_path):
    table = QTable.read(LCS)
    table.meta["population_model"] = "fullpop"
    path = tmp_path / "fullpop.ecsv"
    table.write(path)

    with pytest.raises(SystemExit, match="merger-rate priors differ"):
        paper_results.check_consistency(
            paper_results.load_results(LCS, "ULTRASAT"),
            paper_results.load_results(path, "UVEX"),
        )


def test_the_default_results_directory_is_the_pixelpop_set():
    """The paper is built from PixelPop; FullPop is kept only for comparison."""
    args = paper_results.parse_args([])
    assert args.results_dir == Path("results") / "pixelpop"
    assert args.lcs == args.results_dir / "ultrasat_lcs.ecsv"
    assert args.scenarios == args.results_dir / "observing-scenarios.ecsv"


def test_results_dir_switches_all_four_tables_at_once():
    args = paper_results.parse_args(["--results-dir", "results/fullpop"])
    assert args.uvex == Path("results/fullpop/uvex.ecsv")
    assert args.allsky == Path("results/fullpop/ultrasat_allsky.ecsv")


def test_an_explicit_table_overrides_the_directory():
    args = paper_results.parse_args(["--uvex", "elsewhere/uvex.ecsv"])
    assert args.uvex == "elsewhere/uvex.ecsv"
    assert args.lcs == Path("results/pixelpop/ultrasat_lcs.ecsv")
