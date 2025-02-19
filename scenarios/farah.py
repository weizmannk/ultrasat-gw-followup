#!/usr/bin/env python
"""Convert Farah's (GWTC-3) distribution to suitable format for bayestar-inject and download if necessary."""
# From LIGO DCC : LIGO-T2100512/public/

import os
import sys
import argparse
import requests
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from astropy.table import Table
import logging

from scipy import integrate
from scipy import optimize
from scipy import special
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("farah_processing.log"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)


def parse_arguments():
    """
    Parse command-line arguments.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the ULTRASAT workflow and submit it using HTCondor."
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="./scenarios",
        help="Path to the params file.",
    )
    return parser.parse_args()


def betabinom_k_n(k, n):
    return stats.betabinom(n, k + 1, n - k + 1)


@np.vectorize
def poisson_lognormal_rate_cdf(k, mu, sigma):
    lognorm_pdf = stats.lognorm(s=sigma, scale=np.exp(mu)).pdf

    def func(lam):
        prior = lognorm_pdf(lam)
        poisson_pdf = np.exp(special.xlogy(k, lam) - special.gammaln(k + 1) - lam)
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


def merger_rate(farah_file, ns_max_mass=3, quantiles=[0.05, 0.5, 0.95]):
    """
    Compute astrophysical merger rate.
    """
    # Lower 5% and upper 95% quantiles of log normal distribution
    rates_table = Table(
        [
            # O3 R&P paper Table II row 1 last column
            {"population": "BNS", "lower": 100.0, "mid": 240.0, "upper": 510.0},
            {"population": "NSBH", "lower": 100.0, "mid": 240.0, "upper": 510.0},
            {"population": "BBH", "lower": 100.0, "mid": 240.0, "upper": 510.0},
        ]
    )

    table = Table.read(farah_file)
    source_mass1 = table["mass1"]
    source_mass2 = table["mass2"]
    rates_table["mass_fraction"] = np.array(
        [
            np.sum((source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)),
            np.sum((source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)),
            np.sum((source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)),
        ]
    ) / len(table)

    for key in ["lower", "mid", "upper"]:
        rates_table[key] *= rates_table["mass_fraction"]

    standard_90pct_interval = np.diff(stats.norm.interval(0.9))[0]
    rates_table["mu"] = np.log(rates_table["mid"])
    rates_table["sigma"] = (
        np.log(rates_table["upper"]) - np.log(rates_table["lower"])
    ) / standard_90pct_interval

    lo, mid, hi = poisson_lognormal_rate_quantiles(
        quantiles, rates_table["mu"], rates_table["sigma"]
    )

    return rates_table, (lo, mid, hi)


def download_file(file_url, file_name):
    """Download a file with a progress bar and error handling."""
    if os.path.exists(file_name):
        logging.info(f"File already exists: {file_name}")
        return file_name

    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        file_size = int(response.headers.get("content-length", 0))

        with open(file_name, "wb") as file, tqdm(
            total=file_size, unit="B", unit_scale=True, desc="Downloading"
        ) as progress:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress.update(len(chunk))

        logging.info(f"File downloaded successfully to {file_name}.")
        return file_name
    except requests.RequestException as e:
        logging.error(f"Error downloading the file: {e}")
        return None


if __name__ == "__main__":
    args = parse_arguments()
    output_dir = os.path.abspath(args.outdir)
    os.makedirs(output_dir, exist_ok=True)
    farah_file = os.path.join(output_dir, "farah.h5")

    if not os.path.exists(farah_file):
        file_url = "https://dcc.ligo.org/LIGO-T2100512/public/O1O2O3all_mass_h_iid_mag_iid_tilt_powerlaw_redshift_maxP_events_all.h5"
        file_name = os.path.join(output_dir, file_url.split("/")[-1])
        input_file = download_file(file_url, file_name)

        if input_file is None:
            logging.error("Failed to download the required file.")
            sys.exit(1)

        data = Table.read(input_file)
        Table(
            {
                "mass1": data["mass_1"],
                "mass2": data["mass_2"],
                "spin1z": data["a_1"] * data["cos_tilt_1"],
                "spin2z": data["a_2"] * data["cos_tilt_2"],
            }
        ).write(farah_file, overwrite=True)

        os.remove(input_file)
        logging.info(f"Removed temporary file: {input_file}")
        logging.info(
            f"number of BNS: {len(data[data['mass_1']<3])}, number of NSBH: "
            f"{len(data[(data['mass_1']>=3) & (data['mass_2']<3)])}, number of BBH: "
            f"{len(data[data['mass_2']>=3])}"
        )
        logging.info(f"Processed file saved at: {farah_file}")

    ns_max_mass = 3
    quantiles = [0.05, 0.5, 0.95]
    logging.info("Computing Astrophysical Merger Rate...")
    rate_table, poison_stats = merger_rate(farah_file, ns_max_mass, quantiles)
    logging.info(f"Merger rates computed:\n{rate_table}")

    print(poison_stats)

    # lo = int(np.floor(lo))
    # mid = int(np.round(mid))
    # hi = int(np.ceil(hi))

    # mid, lo, hi = format_with_errorbars(mid, lo, hi)
