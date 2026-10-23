"""Regenerate tests/data/poisson_lognormal_quantiles.json.

Run this ONLY after a deliberate, reviewed change to
``ultrasat_gw_followup.probability.rate.poisson_lognormal_rate_quantiles``:

    python tests/data/make_rate_fixture.py

The regression test tests/test_rate.py::test_poisson_lognormal_quantiles_regression
compares the current output against the committed fixture.
"""

import json

import numpy as np
import scipy

from ultrasat_gw_followup.probability.rate import poisson_lognormal_rate_quantiles

MU, SIGMA = 2.0, 0.5
P = [0.05, 0.16, 0.5, 0.84, 0.95]


def main() -> None:
    quantiles = poisson_lognormal_rate_quantiles(np.asarray(P), MU, SIGMA)
    fixture = {
        "description": (
            "Reference output of "
            "ultrasat_gw_followup.probability.rate.poisson_lognormal_rate_quantiles. "
            "For a log-normal prior on a Poisson rate with mu and sigma, this is "
            "the (continuous) event count k such that the marginal CDF equals "
            "each probability in `p`."
        ),
        "generated_with": {"numpy": np.__version__, "scipy": scipy.__version__},
        "inputs": {"mu": MU, "sigma": SIGMA, "p": P},
        "quantiles": quantiles.tolist(),
    }
    path = "tests/data/poisson_lognormal_quantiles.json"
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
