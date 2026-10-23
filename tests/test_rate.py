"""Unit and regression tests for ultrasat_gw_followup.probability.rate."""

import json
from pathlib import Path

import numpy as np
import pytest

from ultrasat_gw_followup.probability import rate

FIXTURE = Path(__file__).parent / "data" / "poisson_lognormal_quantiles.json"


# --------------------------------------------------------------------------
# format_with_errorbars
# --------------------------------------------------------------------------
def test_format_with_errorbars_float():
    # smallest error bar is 2.0 -> one decimal place
    mid, minus, plus = rate.format_with_errorbars(10.0, 8.0, 13.0)
    assert (mid, minus, plus) == ("10.0", "2.0", "3.0")


def test_format_with_errorbars_integer_inputs_give_no_decimals():
    mid, minus, plus = rate.format_with_errorbars(10, 8, 13)
    assert (mid, minus, plus) == ("10", "2", "3")


def test_format_with_errorbars_subunit_error_adds_precision():
    # smallest error bar is 0.05 -> three decimal places
    mid, minus, plus = rate.format_with_errorbars(1.0, 0.95, 1.2)
    assert (mid, minus, plus) == ("1.000", "0.050", "0.200")


def test_format_with_errorbars_zero_width():
    assert rate.format_with_errorbars(5.0, 5.0, 5.0) == ("5.0", "0", "0")


# --------------------------------------------------------------------------
# betabinom_k_n
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k, n", [(0, 5), (3, 10), (7, 7)])
def test_betabinom_k_n_matches_closed_form_mean(k, n):
    # stats.betabinom(n, a=k+1, b=n-k+1) has mean n * a / (a + b) = n (k+1) / (n+2)
    dist = rate.betabinom_k_n(k, n)
    assert dist.mean() == pytest.approx(n * (k + 1) / (n + 2))
    # support is 0..n and the pmf is a proper distribution
    assert dist.pmf(np.arange(n + 1)).sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# poisson_lognormal_rate_cdf
# --------------------------------------------------------------------------
def test_poisson_lognormal_rate_cdf_is_a_probability_and_monotone():
    k = np.arange(0, 40)
    cdf = rate.poisson_lognormal_rate_cdf(k, 2.0, 0.5)
    assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)
    assert np.all(np.diff(cdf) >= -1e-9)  # non-decreasing in k
    assert cdf[-1] > 0.99  # essentially all mass below k = 39 for these params


def test_cdf_and_quantiles_are_inverses():
    p = np.array([0.1, 0.5, 0.9])
    k = rate.poisson_lognormal_rate_quantiles(p, 2.0, 0.5)
    np.testing.assert_allclose(
        rate.poisson_lognormal_rate_cdf(k, 2.0, 0.5), p, atol=1e-6
    )


# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------
def test_poisson_lognormal_quantiles_regression():
    """Pin the quantile solver against a stored reference.

    The solver combines an adaptive quadrature (scipy.integrate.quad) with a
    bracketing root finder (scipy.optimize.root_scalar). Its output can drift
    at the ~1e-6 level between numpy/scipy versions and platforms, so we compare
    with rtol=1e-4 -- loose enough to absorb that numerical noise, tight enough
    that any real change in the algorithm (which moves values by percent-level
    or more) fails the test. Regenerate the fixture with
    tests/data/make_rate_fixture.py after an intentional change.
    """
    ref = json.loads(FIXTURE.read_text())
    got = rate.poisson_lognormal_rate_quantiles(
        np.asarray(ref["inputs"]["p"]),
        ref["inputs"]["mu"],
        ref["inputs"]["sigma"],
    )
    np.testing.assert_allclose(got, ref["quantiles"], rtol=1e-4)
