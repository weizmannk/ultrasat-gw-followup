#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------------------------------
ABOUT THE SCRIPT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository URL  : https://github.com/weizmannk/ultrasat-gw-followup.git
Creation Date   : January 2024
Description     : This script processes gravitational wave (GW) localization maps to identify events
                  suitable for follow-up observations with ULTRASAT, leveraging its Near Ultraviolet (NUV)
                  observational capabilities.

                  Features:
                  1. Calculates the maximum exposure time (`texp_max`) across the GW localization map for
                     events with localization area less than a specified maximum (`max_area`).
                  2. Filters out events where `texp_max` exceeds a specified maximum exposure time (`max_texp`),
                     adhering to ULTRASAT's observational constraints.
                  3. Outputs CSV files containing the event number and maximum exposure time for the
                     remaining viable events.

                  Assumptions:
                  - Fiducial absolute kilonova (KNe) AB magnitude: -14.1.
                  - Apparent magnitudes are estimated using the mean GW distance (`DISTMEAN`).

                  This script is adapted from https://github.com/criswellalexander/ultrasat-gw-followup.

Usage           : python max-texp-by-sky-loc.py --params /path/to/params.ini --batch_file /path/to/batch_file
"""

import argparse
import configparser
import logging
import os
import sys
from glob import glob

import pandas as pd
import numpy as np
import healpy as hp


from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord, EarthLocation
from astropy.io import fits
from astropy.time import Time
from astropy_healpix import HEALPix

from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import find_greedy_credible_levels

from synphot import ConstFlux1D, SourceSpectrum, exceptions as syn_ex
from m4opt.missions import ultrasat
from m4opt.models import observing

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def estimate_apparent_ABmag(distmean, M_AB=-14.5) -> float:
    """
    Estimate the apparent AB magnitude of a source given its distance and absolute magnitude.

    Args:
        distmean (float): Mean distance to the event in Mpc.
        M_AB (float, optional): Absolute AB magnitude of the source. Defaults to -14.5.

    Returns:
        float: Apparent AB magnitude.
    """
    distmod = 5 * np.log10(distmean * 1e6) - 5  # Distance modulus
    m_AB = M_AB + distmod
    return m_AB


def get_pixel_texp(ipix, nside, obstime, m_obs, band, snr=10.0) -> list:
    """
    Calculate the required exposure time for a specific HEALPix pixel to reach SNR of 10.

    Args:
        ipix (int): HEALPix pixel index.
        nside (int): HEALPix Nside parameter.
        obstime (Time): Observation time.
        m_obs (float): Apparent AB magnitude of the source.
        band (str): Observation band ('nuv' )
        area (float): Telescope collecting area.

    Returns:
        list: [ipix, required exposure time in seconds].
    """
    try:
        # Get pixel sky coordinates
        theta, phi = hp.pix2ang(nside, ipix, nest=True)
        ra = np.rad2deg(phi)
        dec = np.rad2deg(0.5 * np.pi - theta)
        sky_coord = SkyCoord(ICRS(ra=ra * u.deg, dec=dec * u.deg), obstime=obstime)
        observer_location = EarthLocation(0 * u.m, 0 * u.m, 0 * u.m)

        # Calculate exposure time
        with observing(
            observer_location=observer_location,
            target_coord=sky_coord,
            obstime=obstime,
        ):
            if band.upper() in ultrasat.detector.bandpasses.keys():
                texp = ultrasat.detector.get_exptime(
                    snr=snr,
                    source_spectrum=SourceSpectrum(
                        ConstFlux1D, amplitude=m_obs * u.ABmag
                    ),
                    bandpass=band.upper(),
                ).value
            else:
                logging.error(
                    f"Unsupported band requested: {band}. Available bands are: {ultrasat.detector.bandpasses.keys()}"
                )

        return [ipix, texp]
    except Exception as e:
        logging.warning(f"Error calculating exposure for pixel {ipix}: {e}")
        return None


def get_max_texp(
    credible_levels, nside, start_time, m_obs, band, snr, verbose=True
) -> float:
    """
    Calculate the maximum exposure time across the 90% credible region of a GW localization map.

    Args:
        credible_levels (np.ndarray): Credible levels for each pixel.
        nside (int): HEALPix Nside parameter.
        start_time (Time): Observation start time.
        m_obs (float): Apparent AB magnitude of the source.
        band (str): Observation band ('nuv' or 'fuv').
        verbose (bool, optional): If True, print additional information. Defaults to True.

    Returns:
        float: Maximum exposure time required across the 90% credible region.


    """
    try:
        band = band.upper()
        # Calculate exposure times for pixels within the 90% credible region
        texp_list = []
        count = 0
        for ipix, cl in enumerate(credible_levels):
            if cl <= 0.9:
                print("ipix :", ipix)
                pix_texp = get_pixel_texp(ipix, nside, start_time, m_obs, band, snr)
                print(pix_texp)
                if pix_texp is None:
                    count += 1
                    continue
                texp_list.append(pix_texp[1])

        if len(texp_list) == 0:
            logging.warning("No valid pixels found in the 90% credible region.")
            return None

        max_texp = np.max(texp_list)

        if verbose:
            logging.info(f"Failed pixel calculations: {count}")
            logging.info(f"Number of pixels in 90% credible region: {len(texp_list)}")
            logging.info(
                f"Maximum exposure time across 90% localization region: {max_texp:.2f} s"
            )

        return max_texp
    except syn_ex.SynphotError as e:
        logging.error(f"Synphot error: {e}")
        return np.inf


def max_texp_by_sky_loc(
    allsky_dir,
    out_dir,
    batch_file,
    band,
    source_mag,
    dist_measure,
    max_area,
    snr,
    max_texp=7200,
) -> None:
    """
    Main function to determine max exposure times and filter events.

    Performs the following steps:
        1. Determines the maximum exposure time across a GW sky localization map for all events with localization area less than `max_area`.
        2. Discards all events where `texp_max` exceeds `max_texp`.
        3. Saves CSV files with the event number and maximum exposure time.

    Args:
        allsky_dir (str): Path to the directory containing LIGO GW localization skymap files.
        out_dir (str): Output directory (must already exist).
        batch_file (str): Path to CSV describing GW simulations.
        band (str): Observation band ('nuv' or 'fuv').
        source_mag (float): Fiducial kilonova absolute bolometric AB magnitude.
        dist_measure (str): 'mean' or 'upper90' distance estimate for apparent magnitude estimation.
        max_area (float): Maximum localization area in square degrees to consider.
        max_texp (float, optional): Exposure time threshold in seconds. Defaults to 7200 (2 hours).

    Returns:
        None
    """
    try:
        # obstime = Time('2024-01-01 00:00:00')
        if dist_measure == "mean":
            distance_column = "distmean"
        elif dist_measure == "upper90":
            distance_column = "dist(90)"
        else:
            raise ValueError(
                f"Unknown dist_measure '{dist_measure}'. Use 'mean' or 'upper90'."
            )

        # Load events
        events = pd.read_csv(batch_file, delimiter=",")
        events_filtered = events[events["area(90)"] <= max_area]

        if events_filtered.empty:
            logging.info("No events found with area <= max_area.")
            return

        ids = events_filtered["simulation_id"]
        mags = estimate_apparent_ABmag(
            events_filtered[distance_column], M_AB=source_mag
        )
        events_texp = pd.DataFrame(
            {
                "event_id": ids,
                "apparent AB mag": mags,
                "area(90)": events_filtered["area(90)"],
            }
        )

        nside = 64
        nside_hires = nside * 4
        healpix = HEALPix(nside=nside, order="nested", frame=ICRS())
        healpix_hires = HEALPix(nside=nside_hires, order="nested", frame=ICRS())

        fitsfiles = glob(os.path.join(allsky_dir, "*.fits"))
        fitsnames = np.array(
            list(map(lambda filepath: filepath.split("/")[-1], fitsfiles))
        )  # np.array([os.path.basename(filepath) for filepath in fitsfiles])

        rows = []
        ntot = len(ids)
        for progress, file_id in enumerate(ids, start=1):
            ev_name = f"{file_id}.fits"
            logging.info(f"Processing {ev_name} ({progress}/{ntot})")

            if ev_name not in fitsnames:
                logging.warning(f"Event ID {file_id} missing corresponding .fits file.")
                continue

            fitsfile = np.array(fitsfiles)[fitsnames == ev_name]
            if len(fitsfile) != 1:
                logging.warning(
                    f"Warning: duplicate events with ID',{file_id},'; Using first event."
                )
            fitsfile = fitsfile[0]

            # Read sky map and rasterize
            start_time = Time(fits.getval(fitsfile, "DATE-OBS", ext=1))
            skymap_base = read_sky_map(fitsfile, moc=True)["UNIQ", "PROBDENSITY"]
            skymap = rasterize(skymap_base, healpix.level)["PROB"]
            credible_levels = find_greedy_credible_levels(skymap)

            m_obs = events_texp["apparent AB mag"][
                events_texp["event_id"] == int(file_id)
            ].to_numpy()[0]

            print("m_obs :", m_obs)
            print("start_time :", start_time)

            try:

                texp_max = get_max_texp(
                    credible_levels, nside, start_time, m_obs, band, snr, verbose=False
                )
                if texp_max is None:
                    logging.info(
                        f"Invalid skymap for event {file_id}. Increasing NSIDE..."
                    )
                    skymap_hires = rasterize(skymap_base, healpix_hires.level)["PROB"]
                    cls_hires = find_greedy_credible_levels(skymap_hires)
                    texp_max = get_max_texp(
                        cls_hires,
                        nside_hires,
                        start_time,
                        m_obs,
                        band,
                        snr,
                        verbose=False,
                    )
                    if texp_max is None:
                        logging.warning(
                            f"Event {file_id} skipped after failed high-res attempt."
                        )
                        continue
            except syn_ex.SynphotError:
                texp_max = np.inf

            area90 = events_texp.loc[
                events_texp["event_id"] == file_id, "area(90)"
            ].iloc[0]
            rows.append([file_id, m_obs, area90, texp_max])
            logging.info(
                f"Event {file_id}: m_obs={m_obs:.2f}, area(90)={area90:.2f}, texp_max={texp_max:.2f}s"
            )

        events_texp_stats = pd.DataFrame(
            rows, columns=["event_id", "apparent_AB_mag", "area(90)", "texp_max (s)"]
        )

        outname = os.path.basename(batch_file).split("_")[-1].replace(".txt", "")
        output_file = os.path.join(out_dir, f"allsky_texp_max_{band}_{outname}.txt")
        events_texp_stats.to_csv(output_file, index=False, sep=" ")
        logging.info(f"Saved max exposure times to {output_file}")

        events_texp_stats_cut = events_texp_stats[
            events_texp_stats["texp_max (s)"] <= max_texp
        ]
        cut_output_file = os.path.join(
            out_dir, f"allsky_texp_max_cut_{band}_{outname}.txt"
        )
        events_texp_stats_cut.to_csv(cut_output_file, index=False, sep=" ")
        logging.info(f"Saved filtered events to {cut_output_file}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)


def main():
    """
    Main function to execute the max exposure time calculation and event filtering.
    """
    parser = argparse.ArgumentParser(
        description="Calculate required exposure time for BNS follow-up by considering the maximum t_exp across the localization region for events in a batch file."
    )
    parser.add_argument(
        "--params", type=str, required=True, help="Path to the params file"
    )
    parser.add_argument(
        "--batch_file", type=str, required=True, help="Path to the batch file"
    )
    args = parser.parse_args()

    # Load configuration
    config = configparser.ConfigParser()
    if not os.path.isfile(args.params):
        logging.error(f"Params file '{args.params}' does not exist.")
        sys.exit(1)
    config.read(args.params)

    try:
        obs_scenario_dir = config.get("params", "obs_scenario")
        out_dir = config.get("params", "save_directory")
        band = config.get("params", "band")
        source_mag = config.getfloat("params", "KNe_mag_AB")
        dist_measure = config.get("params", "distance_measure")
        max_texp = config.getfloat("params", "max_texp", fallback=7200)
        max_area = config.getfloat("params", "max_area", fallback=200)
        snr = config.getfloat("params", "snr")
        logging.info("Configuration parameters loaded successfully.")
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        logging.error(f"Error reading configuration: {e}")
        sys.exit(1)

    allsky_dir = os.path.join(obs_scenario_dir, "allsky")
    texp_out_dir = os.path.join(out_dir, "texp_out")

    # Create output directory if it doesn't exist
    os.makedirs(texp_out_dir, exist_ok=True)

    # Run the main processing function
    max_texp_by_sky_loc(
        allsky_dir=allsky_dir,
        out_dir=texp_out_dir,
        batch_file=args.batch_file,
        band=band,
        source_mag=source_mag,
        dist_measure=dist_measure,
        max_area=max_area,
        snr=snr,
        max_texp=max_texp,
    )


if __name__ == "__main__":
    main()
