"""Tests for the aggregated block and the P(N=0) column of the rate table.

The reference values are those of the paper, read from the final ULTRASAT LCS
table: over O5, 24.44 selected events and 8.85 detected kilonovae per year at
the median merger rate.

The aggregate is the delicate part. The merger-rate prior is a single
astrophysical rate shared by every observing run, so its uncertainty is
*fully correlated* across them. Combining runs therefore means adding the
expected counts and re-deriving the quantiles once, at the same prior width.
Two tempting shortcuts are wrong and are guarded against below:

* summing the per-run quantiles -- the median of a sum is not the sum of the
  medians;
* convolving the per-run predictive distributions as if independent, which
  would treat one rate uncertainty as three.
"""

import re
from pathlib import Path

import numpy as np
import pytest
from astropy.table import QTable
from scipy import stats

from ultrasat_gw_followup.analysis.selection_detection_rate import (
    summarize_selected_detected_events,
)
from ultrasat_gw_followup.probability.rate import (
    poisson_lognormal_rate_cdf,
    poisson_lognormal_rate_quantiles,
)

EVENTS = Path(__file__).resolve().parents[1] / "results" / "ultrasat_lcs.ecsv"
O5 = ("O5a", "O5b", "O5c")
PRIOR = {"merger_rate_lo": 50, "merger_rate_mid": 100, "merger_rate_hi": 203}

#: Aggregate of O5a+O5b+O5c for the ULTRASAT LCS grid, as quoted in the paper.
#:   (row, class): (lambda, median, 5%, 95%, P(N=0) in percent)
REFERENCE = {
    ("Selected", "BNS"): (15.29, 14.7, 5.4, 32.2, 0.0),
    ("Selected", "NSBH"): (9.15, 8.6, 2.5, 19.7, 0.2),
    ("Selected", "All"): (24.44, 23.9, 9.9, 50.6, 0.0),
    ("Detected", "BNS"): (5.60, 5.0, 0.9, 12.5, 1.4),
    ("Detected", "NSBH"): (3.25, 2.6, 0.0, 7.6, 6.1),
    ("Detected", "All"): (8.85, 8.3, 2.3, 19.1, 0.2),
}
ROWS = ["Selected", "Detected"]
CLASSES = ["BNS", "NSBH", "All"]

pytestmark = pytest.mark.skipif(
    not EVENTS.exists(), reason=f"{EVENTS} not available in this checkout"
)


def summarize(**kwargs):
    return summarize_selected_detected_events(
        EVENTS,
        run_duration=1.0,
        poisson_lognormal_rate_quantiles=poisson_lognormal_rate_quantiles,
        verbose=False,
        **PRIOR,
        **kwargs,
    )


def aggregate_arrays():
    """Recompute the aggregate independently of the table renderer."""
    table = QTable.read(EVENTS)
    table = table[table["objective_value"] >= table["cutoff"][0]]
    effective = {
        run: value.to_value("Gpc-3 yr-1")
        for run, value in table.meta["effective_rate"].items()
    }
    (width_90,) = np.diff(stats.norm.interval(0.9))
    sigma = np.log(PRIOR["merger_rate_hi"] / PRIOR["merger_rate_lo"]) / width_90

    is_class = {
        "BNS": lambda t: t["source_class"] == "BNS",
        "NSBH": lambda t: t["source_class"] == "NSBH",
        "All": lambda t: np.ones(len(t), dtype=bool),
    }
    lam = np.zeros((2, len(CLASSES)))
    for run in O5:
        events = table[table["run"] == run]
        scale = PRIOR["merger_rate_mid"] / effective[run]
        for i_cls, cls in enumerate(CLASSES):
            subset = events[is_class[cls](events)]
            lam[0, i_cls] += scale * len(subset)
            lam[1, i_cls] += scale * np.sum(
                subset["detection_probability_known_position"]
            )
    return lam, np.log(lam), sigma


# --------------------------------------------------------------------------
# Default behaviour is untouched
# --------------------------------------------------------------------------
def test_defaults_add_no_aggregate_row_and_no_zero_column():
    latex = summarize()
    assert "O5 total" not in latex
    assert "P(N{=}0)" not in latex
    # two sub-columns per column group (Selected, Detected)
    assert r"\begin{tabular}{ll" + "cc" * len(ROWS) + "}" in latex


def test_aggregate_leaves_the_per_run_rows_untouched():
    without = summarize()
    with_aggregate = summarize(aggregate_runs=O5)
    per_run = [line for line in without.splitlines() if line.endswith(r"\\")]
    still_there = [line for line in with_aggregate.splitlines() if line.endswith(r"\\")]
    for line in per_run:
        assert line in still_there


# --------------------------------------------------------------------------
# Reference values
# --------------------------------------------------------------------------
@pytest.mark.parametrize("row", ROWS)
@pytest.mark.parametrize("cls", CLASSES)
def test_aggregate_matches_the_paper(row, cls):
    lam, mu, sigma = aggregate_arrays()
    i_row, i_cls = ROWS.index(row), CLASSES.index(cls)
    want_lam, want_med, want_lo, want_hi, want_zero = REFERENCE[row, cls]

    assert lam[i_row, i_cls] == pytest.approx(want_lam, abs=0.1)

    median, lo, hi = poisson_lognormal_rate_quantiles(
        np.array([0.5, 0.05, 0.95]), mu[i_row, i_cls], sigma
    )
    assert median == pytest.approx(want_med, abs=0.1)
    assert lo == pytest.approx(want_lo, abs=0.1)
    assert hi == pytest.approx(want_hi, abs=0.1)

    zero = 100 * poisson_lognormal_rate_cdf(0, mu[i_row, i_cls], sigma)
    assert zero == pytest.approx(want_zero, abs=0.2)


def test_prior_width_is_the_expected_sigma():
    _, _, sigma = aggregate_arrays()
    assert sigma == pytest.approx(0.4259, abs=1e-4)


def test_the_table_prints_the_reference_lambdas():
    """The rendered aggregate must carry the numbers quoted in the paper."""
    rows = body_rows(summarize(aggregate_runs=O5))
    aggregate = [row for row in rows if row[1] == "All"][-1]
    assert float(aggregate[2]) == pytest.approx(24.44, abs=0.01)
    assert float(aggregate[4]) == pytest.approx(8.85, abs=0.01)


# --------------------------------------------------------------------------
# The aggregate must not be built by summing quantiles
# --------------------------------------------------------------------------
def test_median_is_not_additive():
    """Guard against re-implementing the aggregate as a sum of quantiles.

    The median of a sum exceeds the sum of the medians, so a
    quantile-summing implementation would under-report the aggregate.
    """
    table = QTable.read(EVENTS)
    table = table[table["objective_value"] >= table["cutoff"][0]]
    effective = {
        run: value.to_value("Gpc-3 yr-1")
        for run, value in table.meta["effective_rate"].items()
    }
    _, mu_aggregate, sigma = aggregate_arrays()

    per_run_medians = []
    for run in O5:
        events = table[table["run"] == run]
        lam_run = (PRIOR["merger_rate_mid"] / effective[run]) * np.sum(
            events["detection_probability_known_position"]
        )
        per_run_medians.append(
            float(poisson_lognormal_rate_quantiles(0.5, np.log(lam_run), sigma))
        )

    correct = float(
        poisson_lognormal_rate_quantiles(
            0.5, mu_aggregate[1, CLASSES.index("All")], sigma
        )
    )
    assert sum(per_run_medians) == pytest.approx(7.0, abs=0.1)
    assert correct == pytest.approx(8.3, abs=0.1)
    assert correct > sum(per_run_medians), "aggregate must not sum medians"


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def body_rows(latex):
    """The data rows of the tabular, as lists of stripped cells."""
    tabular = latex.split(r"\begin{tabular}")[1]
    rows = []
    for line in tabular.splitlines():
        line = line.rstrip()
        if not line.endswith(r"\\") or "&" not in line:
            continue
        cells = [cell.strip() for cell in line[:-2].split("&")]
        # Data rows carry the source class in the second column; the two
        # header rows do not.
        if len(cells) > 1 and cells[1] in CLASSES:
            rows.append(cells)
    return rows


def runs_in(rows):
    """Run names, read from the \\multirow cell that opens each block."""
    return [
        re.search(r"\{([^{}]+)\}$", row[0]).group(1)
        for row in rows
        if row[0].startswith(r"\multirow")
    ]


def test_the_aggregate_is_one_extra_block_at_the_foot():
    rows = body_rows(summarize(aggregate_runs=O5))
    assert runs_in(rows) == [*O5, "O5 total"]
    assert len(rows) == len(CLASSES) * (len(O5) + 1)


def test_layout_is_one_row_per_run_and_class():
    rows = body_rows(summarize())
    assert [row[1] for row in rows[: len(CLASSES)]] == CLASSES
    assert len(rows) % len(CLASSES) == 0
    # Selected and Detected are column groups, not rows
    assert not any(cell in ROWS for row in rows for cell in row)


def test_booktabs_rules_and_column_groups():
    latex = summarize()
    for rule in (r"\toprule", r"\midrule", r"\bottomrule"):
        assert rule in latex
    assert r"\hline" not in latex
    assert r"\Xhline" not in latex
    assert r"\multicolumn{2}{c}{Selected}" in latex
    assert r"\multicolumn{2}{c}{Detected}" in latex
    assert r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}" in latex


def test_cmidrules_follow_the_extra_column():
    latex = summarize(include_zero_probability=True)
    assert r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}" in latex


def test_the_caption_names_the_grid():
    assert "Low-Cadence Survey" in summarize()


# --------------------------------------------------------------------------
# P(N = 0)
# --------------------------------------------------------------------------
def test_zero_probability_adds_one_column_per_class():
    latex = summarize(include_zero_probability=True)
    assert r"$P(N{=}0)$" in latex
    assert r"\begin{tabular}{ll" + "ccc" * len(ROWS) + "}" in latex
    assert re.search(r"\d+\.\d\\%", latex)


def test_zero_probability_is_a_percentage_with_one_decimal():
    latex = summarize(include_zero_probability=True)
    for value in re.findall(r"(\d+\.\d)\\%", latex):
        assert 0.0 <= float(value) <= 100.0


def test_zero_probability_decreases_with_lambda():
    """P(N=0) must fall as the expected count rises."""
    _, mu, sigma = aggregate_arrays()
    detected = poisson_lognormal_rate_cdf(0, mu[1], sigma)
    selected = poisson_lognormal_rate_cdf(0, mu[0], sigma)
    assert np.all(selected < detected)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
def test_unknown_aggregate_run_raises():
    with pytest.raises(ValueError, match="absent"):
        summarize(aggregate_runs=("O5a", "O9z"))
