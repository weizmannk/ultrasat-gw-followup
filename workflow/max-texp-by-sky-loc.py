#!/usr/bin/env python3
"""
ULTRASAT GW Localization Analysis Script

This script analyzes gravitational wave (GW) localization maps to identify events
suitable for ULTRASAT follow-up observations in the Near Ultraviolet (NUV).

Key Features:
1. Computes maximum exposure time (`texp_max`) for events with localization area < `max_area`.
2. Filters out events exceeding ULTRASAT’s exposure constraints (`max_texp`).
3. Outputs `.csv` files listing viable events with `texp_max`.

Assumptions:
- Fiducial kilonova absolute magnitude: e.g = -16 ABmag for ULTRASAT (GW170817).
- Apparent magnitudes estimated using mean GW distance (`DISTMEAN`).

Usage:
    python3 max-texp-by-sky-loc.py --params /path/to/params.ini --batch_file /path/to/batch_file
"""

import argparse
import configparser
import logging

import os
import sys
from glob import glob
import pandas as pd
import numpy as np

from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord, EarthLocation, Distance
from astropy.io import fits
from astropy.time import Time

from astropy_healpix import HEALPix
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import find_greedy_credible_levels
from ligo.skymap import distance
from scipy import stats

from m4opt.utils.console import status
from m4opt.missions import ultrasat
from m4opt.synphot import observing, TabularScaleFactor
from m4opt.synphot.extinction import DustExtinction
import synphot


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def estimate_apparent_ABmag(distmean, M_AB=-14.5) -> float:
    """
    Estimate the apparent AB magnitude of a source given its distance and absolute magnitude.
    """
    distmod = 5 * np.log10(distmean * 1e6) - 5
    return M_AB + distmod


def gaussian_mag(skymap_flat, absmag_mean, absmag_stdev):

    distmean, diststd, _ = distance.parameters_to_moments(
        skymap_flat["DISTMU"],
        skymap_flat["DISTSIGMA"],
    )

    logdist_sigma2 = np.log1p(np.square(diststd / distmean))
    logdist_sigma = np.sqrt(logdist_sigma2)
    logdist_mu = np.log(distmean) - 0.5 * logdist_sigma2
    a = 5 / np.log(10)
    appmag_mu = absmag_mean + a * logdist_mu + 25
    appmag_sigma = np.sqrt(np.square(absmag_stdev) + np.square(a * logdist_sigma))
    quantiles = np.linspace(0.05, 0.95, 5)
    appmag_quantiles = stats.norm(
        loc=appmag_mu[:, np.newaxis], scale=appmag_sigma[:, np.newaxis]
    ).ppf(quantiles)

    return appmag_quantiles


def get_max_texp_by_sky_loc(
    allsky_dir,
    texp_out_dir,
    batch_file,
    bandpass,
    absmag_mean,
    absmag_stdev,
    dist_measure,
    max_area,
    snr,
    max_texp=7200,
    delay="6 hr",
    nside=128,
):
    try:
        # Validate distance measure
        distance_column = {"mean": "distmean", "upper90": "dist(90)"}.get(dist_measure)

        if not distance_column:
            raise ValueError(
                f"Unknown dist_measure '{dist_measure}'. Use 'mean' or 'upper90'."
            )

        # Load events and filter by max area
        events = pd.read_csv(batch_file, delimiter=",")
        events_filtered = events[events["area(90)"] <= max_area]

        if events_filtered.empty:
            logging.info(f"No events found with area <= {max_area}.")
            return

        ids = events_filtered["simulation_id"]
        skymap_files = glob(os.path.join(allsky_dir, "*.fits"))
        fitsnames = {os.path.basename(f): f for f in skymap_files}

        rows = []
        for progress, file_id in enumerate(ids, start=1):
            ev_name = f"{file_id}.fits"
            logging.info(f"Processing {ev_name} ({progress}/{len(ids)})")

            skymap_file = fitsnames.get(ev_name)
            if not skymap_file:
                logging.warning(
                    f"Missing .fits file for Event ID {file_id}. Skipping..."
                )
                continue

            try:
                with status(f"Loading sky map for event {file_id}"):
                    hpx = HEALPix(nside, frame=ICRS(), order="nested")
                    skymap_moc = read_sky_map(skymap_file, moc=True)
                    skymap_flat = rasterize(skymap_moc, hpx.level)
                    event_time = Time(
                        Time(skymap_moc.meta["gps_time"], format="gps").utc,
                        format="iso",
                    )

                with status(f"Propagating orbit for event {file_id}"):
                    obstime = event_time + u.Quantity(delay)
                    observer_location = ultrasat.observer_location(obstime)

                with status(f"Computing credible region for event {file_id}"):
                    # Compute the credible levels for each pixel in the sky map by sorting
                    # probabilities in descending order and calculating their cumulative sum.
                    # This helps define regions corresponding to specific confidence levels (in this case, 90%).
                    credible_levels = find_greedy_credible_levels(skymap_flat["PROB"])
                    # selected_pixels = np.array([ipix for ipix, cl in enumerate(credible_levels) if cl <= 0.9])
                    selected_pixels = np.where(credible_levels <= 0.9)[0]

                    if not selected_pixels.size:
                        logging.info(
                            f"No valid pixels found in 90% credible region for Event {file_id}. Skipping..."
                        )
                        continue

                    if np.max(selected_pixels) >= 12 * hpx.nside**2:
                        hpx = HEALPix(hpx.nside * 2, frame=ICRS(), order="nested")
                        selected_pixels = selected_pixels[
                            selected_pixels < 12 * hpx.nside**2
                        ]

                        # Re-check after increasing NSIDE
                        if not selected_pixels.size:
                            logging.info(
                                f"Event {file_id} still has an invalid skymap after increasing NSIDE. Skipping..."
                            )
                            continue

                    # FIXME , check if we nned np.arange(hpx.npix)  or selected_pixels
                    sky_coords = hpx.healpix_to_skycoord(selected_pixels)
                    skymap_selected = skymap_flat[selected_pixels]

                with status(f"Estimating apparent AB magnitude for event {file_id}"):
                    distmod = Distance(skymap_moc.meta["distmean"] * u.Mpc).distmod
                    mag_obs = (absmag_mean * u.ABmag + distmod).value

                with status(f"Computing exposure time for event {file_id}"):
                    exptime_pixel_s = compute_exptime_pixel_s(
                        skymap_selected,
                        obstime,
                        observer_location,
                        sky_coords,
                        absmag_mean,
                        absmag_stdev,
                        bandpass,
                        mag_obs,
                        snr,
                    )

                    if not exptime_pixel_s.size:
                        logging.warning(
                            f"No valid exposure times found for Event {file_id}. Skipping..."
                        )
                        continue

                    exptime_max_s = np.max(exptime_pixel_s)

            except (synphot.exceptions.SynphotError, KeyError) as e:
                logging.error(f"Error processing Event {file_id}: {e}")
                continue

            rows.append(
                {
                    "event_id": file_id,
                    "apparent_AB_mag": mag_obs,
                    "texp_max (s)": exptime_max_s,
                }
            )

        if not rows:
            logging.info("No valid events processed.")
            return

        events_texp_stats = pd.DataFrame(
            rows, columns=["event_id", "apparent_AB_mag", "texp_max (s)"]
        )

        outname = os.path.basename(batch_file).split("_")[-1].replace(".csv", "")
        output_file = os.path.join(
            texp_out_dir, f"allsky_texp_max_{bandpass}_{outname}.csv"
        )
        events_texp_stats.to_csv(output_file, index=False)
        logging.info(f"Saved max exposure times to {output_file}")

        events_texp_stats_cut = events_texp_stats[
            events_texp_stats["texp_max (s)"] <= max_texp
        ]
        cut_output_file = os.path.join(
            texp_out_dir, f"allsky_texp_max_cut_{bandpass}_{outname}.csv"
        )
        events_texp_stats_cut.to_csv(cut_output_file, index=False)
        logging.info(f"Saved filtered events to {cut_output_file}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)


def compute_exptime_pixel_s(
    skymap,
    obstime,
    observer_location,
    sky_coords,
    absmag_mean,
    absmag_stdev,
    bandpass,
    mag_obs,
    snr=10,
):
    """
    Compute exposure time per pixel, only considering pixels within the 90% credible region.
    """
    # absmag_stdev= None
    target_quantile = 0.5  # 50% confidence level
    quantiles = np.linspace(0.05, 0.95, 5)
    index = np.argmin(np.abs(quantiles - target_quantile))

    with status("Evaluating exposure time map"):
        # Compute exposure time for selected pixels
        with observing(
            observer_location=observer_location,
            target_coord=sky_coords[:, np.newaxis],
            obstime=obstime,
        ):
            if bandpass.upper() not in ultrasat.detector.bandpasses:
                raise ValueError(
                    f"Bandpass '{bandpass.upper()}' not found in ULTRASAT detector. Cannot compute exposure time."
                )

            # Select source spectrum based on whether magnitude uncertainty is provided
            if absmag_stdev is not None:
                appmag_quantiles = gaussian_mag(skymap, absmag_mean, absmag_stdev)
                source_spectrum = (
                    synphot.SourceSpectrum(synphot.ConstFlux1D(0 * u.ABmag))
                    * synphot.SpectralElement(
                        TabularScaleFactor(
                            (
                                appmag_quantiles * u.mag(u.dimensionless_unscaled)
                            ).to_value(u.dimensionless_unscaled)
                        )
                    )
                    * DustExtinction()
                )

            else:
                source_spectrum = (
                    synphot.SourceSpectrum(synphot.ConstFlux1D(mag_obs * u.ABmag))
                    * DustExtinction()
                )

            # Compute exposure time
            exptime_pixel_s = ultrasat.detector.get_exptime(
                snr=snr,
                source_spectrum=source_spectrum,
                bandpass=bandpass.upper(),
            ).to_value(u.s)

            # If using quantiles, extract the 50% confidence level
            if absmag_stdev is not None:
                exptime_pixel_s = exptime_pixel_s[:, index]

    return exptime_pixel_s


def main():
    """Main function to execute the max exposure time calculation and event filtering."""
    parser = argparse.ArgumentParser(
        description="Calculate required exposure time for BNS follow-up by considering the maximum t_exp across the localization region."
    )
    parser.add_argument(
        "--params", type=str, required=True, help="Path to the params file"
    )
    parser.add_argument(
        "--batch_file", type=str, required=True, help="Path to the batch file"
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    if not os.path.isfile(args.params):
        logging.error(f"Params file '{args.params}' does not exist.")
        sys.exit(1)
    config.read(args.params)

    try:
        obs_scenario_dir = config.get("params", "obs_scenario")
        out_dir = config.get("params", "save_directory")
        bandpass = config.get("params", "bandpass")
        absmag_mean = config.getfloat("params", "absmag_mean")
        absmag_stdev = config.getfloat("params", "absmag_stdev")
        dist_measure = config.get("params", "distance_measure")
        max_texp = config.getfloat("params", "max_texp", fallback=720)
        max_area = config.getfloat("params", "max_area", fallback=200)
        snr = config.getfloat("params", "snr")
        delay = config.get("params", "delay")
        nside = config.getint("params", "nside")
        logging.info("Configuration parameters loaded successfully.")
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        logging.error(f"Error reading configuration: {e}")
        sys.exit(1)

    allsky_dir = os.path.join(obs_scenario_dir, "allsky")
    texp_out_dir = os.path.join(out_dir, "texp_out")
    os.makedirs(texp_out_dir, exist_ok=True)

    logging.info("Starting max exposure time calculation...")

    get_max_texp_by_sky_loc(
        allsky_dir=allsky_dir,
        texp_out_dir=texp_out_dir,
        batch_file=args.batch_file,
        bandpass=bandpass,
        absmag_mean=absmag_mean,
        absmag_stdev=absmag_stdev,
        dist_measure=dist_measure,
        max_area=max_area,
        snr=snr,
        max_texp=max_texp,
        delay=delay,
        nside=nside,
    )


if __name__ == "__main__":
    main()
