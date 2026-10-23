# """
# Functions for propagating errors in rates.

# Reproduced from
# https://github.com/lpsinger/observing-scenarios-simulations/blob/main/plots-and-tables.ipynb.
# """

import numpy as np
from scipy import integrate, optimize, special, stats


def betabinom_k_n(k, n):
    """Beta-binomial probability mass at k for n trials."""
    return stats.betabinom(n, k + 1, n - k + 1)


@np.vectorize
def poisson_lognormal_rate_cdf(k, mu, sigma):
    """Marginalized CDF of Poisson count k with log-normal prior on the rate."""
    lognorm_pdf = stats.lognorm(s=sigma, scale=np.exp(mu)).pdf

    def func(lam):
        prior = lognorm_pdf(lam)
        # poisson_pdf = np.exp(special.xlogy(k, lam) - special.gammaln(k + 1) - lam)
        poisson_cdf = special.gammaincc(k + 1, lam)
        return poisson_cdf * prior

    # Marginalize over lambda.
    #
    # Note that we use scipy.integrate.odeint instead
    # of scipy.integrate.quad because it is important for the stability of
    # root_scalar below that we calculate the pdf and the cdf at the same time,
    # using the same exact quadrature rule.
    cdf, _ = integrate.quad(func, 0, np.inf, epsabs=0)
    return cdf


@np.vectorize
def poisson_lognormal_rate_quantiles(p, mu, sigma):
    """Find the quantiles of a Poisson distribution with
    a log-normal prior on its rate.

    Parameters
    ----------
    p : float
        The quantiles at which to find the number of counts.
    mu : float
        The mean of the log of the rate.
    sigma : float
        The standard deviation of the log of the rate.

    Returns
    -------
    k : float
        The number of events.

    Notes
    -----
    This algorithm treats the Poisson count k as a continuous
    real variable so that it can use the scipy.optimize.root_scalar
    root finding/polishing algorithms.
    """

    def func(k):
        return poisson_lognormal_rate_cdf(k, mu, sigma) - p

    if func(0) >= 0:
        return 0

    result = optimize.root_scalar(func, bracket=[0, 1e6])
    return result.root


def format_with_errorbars(mid, lo, hi):
    """Format a value with asymmetric error bars for display."""
    plus = hi - mid
    minus = mid - lo
    smallest = min(max(0, plus), max(0, minus))

    if smallest == 0:
        return str(mid), "0", "0"
    decimals = 1 - int(np.floor(np.log10(smallest)))

    if all(np.issubdtype(type(_), np.integer) for _ in (mid, lo, hi)):
        decimals = min(decimals, 0)

    plus, minus, mid = np.round([plus, minus, mid], decimals)
    if decimals > 0:
        fstring = f"%.0{decimals}f"
    else:
        fstring = "%d"
    return [fstring % _ for _ in [mid, minus, plus]]
