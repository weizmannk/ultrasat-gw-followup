#!/usr/bin/env python
# coding: utf-8

"""
This is a script to look at uvex-followup results and determine the overall success rate of triggered observations, as well as collect information as to the time of first detection.

Run via
 > python3 followup_success.py /path/to/run/output/directory/ --prior_min [KN intrinsic magnitude prior minimum] --prior_max [KN intrinsic magnitude prior maximum]

Produces a file called "allsky_success.txt" in the original run directory. This file contains various quantities related to the success of the scheduled observations.

Note: overall success rate assumes a uniform prior on the intrinsic kilonova absolute AB magnitude s.t. p(M_AB_KN) ~ U(prior_min,prior_max)

"""

from astropy import units as u
from ligo.skymap.tool import ArgumentParser, FileType
from dorado.scheduling import mission as _mission
from dorado.scheduling.units import equivalencies

from astropy.coordinates import ICRS
from astropy_healpix import HEALPix, lonlat_to_healpix
from astropy.io import fits
from astropy.time import Time
from astropy.table import QTable
from ligo.skymap.io import read_sky_map
from ligo.skymap.bayestar import rasterize
from ligo.skymap import plot
from ligo.skymap.postprocess import find_greedy_credible_levels
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import sys, os, configparser
import argparse
import pandas as pd


## adapting the compute_tiling() function from uvex-followup
def compute_success(
    allsky_select,
    injections,
    fitsloc,
    schedloc,
    rundir,
    assumed_mag,
    dist_measure,
    prior_min=-16.5,
    prior_max=-12.1,
):
    """
    Function to compute the number of UVEX pointings needed to tile a GW localization region.

    Arguments
    -----------------------------
    allsky_select (str)   : /path/to/allsky_selected_gwem.txt
    injections (str)      : /path/to/injections.dat
    fitsloc (str)         : /path/to/directory/with/skymaps/
    schedloc (str)        : /path/to/directory/with/schedules/ (as computed by the uvex scheduler)
    rundir (str)          : /path/to/save/directory/
    assumed_mag (float)   : assumed absolute KN magnitude
    dist_measure (str)    : Distance estimate used ('mean' or 'upper90')
    prior_min (float)     : Intrinsic KN Magnitude prior minimum (assumed uniform). Default: -16.5 AB mags
    prior_max (float)     : Intrinsic KN Magnitude prior maximum (assumed uniform). Default: -12.1 AB mags
    """
    ## manually define some arguments
    mission = getattr(_mission, "uvex")
    #     mission = _mission.uvex()
    nside = 64

    if dist_measure == "mean":
        distance_column = "distmean"
    elif dist_measure == "upper90":
        raise ValueError(
            "The upper90 distance measure is not supported for this script."
        )
    #         distance_column = 'dist(90)'

    healpix = HEALPix(nside, order="nested", frame=ICRS())

    ## get event numbers
    event_df = pd.read_csv(allsky_select, delimiter=",")  # ,skiprows=1)

    event_list = event_df["event_id"].tolist()

    texp_list = event_df["texp_sched (ks)"].tolist()

    distmeans = event_df[distance_column].to_numpy() * 1e6

    ## get paths to data file directories
    fitslist = list(map(lambda ev_num: str(ev_num) + ".fits", event_list))
    schedlist = list(map(lambda ev_num: str(ev_num) + ".ecsv", event_list))

    ## grab injections
    inj_df = pd.read_csv(injections, sep="\t")

    id_filt = [
        True if inj_df["simulation_id"][i] in event_list else False
        for i in range(len(inj_df))
    ]

    inj_skylocs = inj_df[id_filt][
        ["longitude", "latitude"]
    ].to_numpy()  ## array of long, lat for every selected event, for events in event_list
    inj_distances = (
        inj_df[id_filt]["distance"].to_numpy() * 10**6
    )  ## array of d _L_inj in pc for every selected event, for events in event_list

    rows = []

    ## loop over .fits and .ecsv files
    for ii, (event, fitsfile, schedule, texp) in enumerate(
        zip(event_list, fitslist, schedlist, texp_list)
    ):

        # Read multi-order sky map and rasterize to working resolution
        skymap = read_sky_map(fitsloc + fitsfile, moc=True)["UNIQ", "PROBDENSITY"]
        skymap = rasterize(skymap, healpix.level)["PROB"]
        # check to see if file is empty before loading because there are empty files for some reason
        if os.stat(schedloc + schedule).st_size == 0:
            print("Error: empty schedule for event", event)
            continue
        schedule = QTable.read(schedloc + schedule, format="ascii.ecsv")

        ## get the injected position
        lon = inj_skylocs[ii, 0] * u.rad
        lat = inj_skylocs[ii, 1] * u.rad
        inj_pix_idx = lonlat_to_healpix(lon, lat, nside, order="nested")

        indices = np.asarray([], dtype=np.intp)

        row_count = 0
        t_first_detection = np.nan
        success_loc = False  # whether we observed the pixel with the counterpart

        ## for every tile, check if the injected location is in the field
        for row in schedule:
            row_count += 1
            new_indices = mission.fov.footprint_healpix(
                healpix, row["center"], row["roll"]
            )
            if inj_pix_idx in new_indices:
                t_first_detection = row_count * texp
                success_loc = True
                break

        if success_loc:
            print(
                "Injected location of event",
                event,
                "was successfully observed; time to first detection:",
                t_first_detection,
                "ks",
            )
        else:
            print("Event", event, "was not successful.")

        ## now check if we achieved sufficient depth to observe the counterpart
        ## assuming a uniform prior on the absolute magnitude of U(-16.5,-12.1)

        ## get the observed apparent limiting magnitude
        observed_lim_mag = assumed_mag + 5 * np.log10(distmeans[ii]) - 5
        print("dist mod ", 5 * np.log10(distmeans[ii]) - 5)
        print("obs lim mag ", observed_lim_mag)
        ## injected distance modulus
        dist_mod_inj = 5 * np.log10(inj_distances[ii]) - 5
        print("dist mod inj ", dist_mod_inj)
        ## integral of uniform prior CDF from brightest (smallest) m_AB_true up to the observed limiting magnitude
        ## this is linear so we can compute it exactly:
        mod_prior_min = prior_min + dist_mod_inj
        mod_prior_max = prior_max + dist_mod_inj
        print("mod prior min ", mod_prior_min)
        print("mod prior max ", mod_prior_max)
        integrated_prob = (observed_lim_mag - mod_prior_min) / (
            mod_prior_max - mod_prior_min
        )

        print("pre-norm integrated prob ", integrated_prob)
        ## catch cases where the integrated probability is > 1
        if integrated_prob > 1.0:
            if distmeans[ii] < inj_distances[ii]:
                print("inj dist", inj_distances[ii] / 1e6)
                print("mpmin", mod_prior_min)
                print("mpmax", mod_prior_max)
                print("distmean", distmeans[ii] / 1e6)
                print("limmag_obs", observed_lim_mag)
                print("prob", integrated_prob)
                raise ValueError(
                    "Probability of observation is greater than 1, but we didn't overestimate the distance. This shouldn't happen outside of a bug."
                )
            integrated_prob = 1.0
        elif integrated_prob < 0.0:
            if prior_min > prior_max:
                raise ValueError(
                    "prior_min > prior_max. The prior minimum must be less than the prior maximum."
                )
            else:
                ## this means our assumed apparent magnitude is brighter than the brightest apparent magnitude allowed by the prior; p(success) = 0.
                integrated_prob = 0.0
        print("integrated prob ", integrated_prob)
        #         print('inj dist',inj_distances[ii]/1e6)
        #         print('mpmin',mod_prior_min)
        #         print('mpmax',mod_prior_max)
        #         print('distmean',distmeans[ii]/1e6)
        #         print('limmag_obs',observed_lim_mag)
        #         print('prob',integrated_prob)
        #         import pdb; pdb.set_trace()

        total_p_obs = success_loc * integrated_prob

        rows.append(
            [
                event,
                success_loc,
                integrated_prob,
                total_p_obs,
                t_first_detection,
                texp,
                lon,
                lat,
                inj_pix_idx,
            ]
        )

    ## Format and save
    events_success = pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "success_skyloc",
            "p_obs_depth",
            "p_obs_tot",
            "t_first (ks)",
            "texp_sched (ks)",
            "lon_inj",
            "lat_inj",
            "pix_inj",
        ],
    )

    ## get success rate
    success_rate = events_success["p_obs_tot"].to_numpy().sum() / len(events_success)
    print(
        "Success rate (fraction of observations that found the true counterpart) is",
        success_rate,
    )

    events_success.to_csv(rundir + "/allsky_success.txt", index=False, sep=" ")

    ## quick plot
    plt.figure()
    plt.hist(events_success["t_first (ks)"] * 1000 / 3600, bins=20)
    plt.xlabel("Time to First Detection (hrs)")
    plt.savefig(rundir + "/time_to_detection_plot.png", dpi=200)
    plt.close()

    return


if __name__ == "__main__":

    ## set up argparser
    parser = argparse.ArgumentParser(
        description="Get success rate in post for a uvex-followup run."
    )
    parser.add_argument("rundir", type=str, help="/path/to/original/run/directory/")

    parser.add_argument(
        "--prior_min",
        type=float,
        help="KN intrinsic magnitude prior minimum in AB mags",
        default=-16.5,
    )
    parser.add_argument(
        "--prior_max",
        type=float,
        help="KN intrinsic magnitude prior maximum in AB mags",
        default=-12.1,
    )

    args = parser.parse_args()

    ## get required arguments
    original_params = args.rundir + "/params.ini"
    allsky_select = args.rundir + "/allsky_selected_gwem.txt"
    schedloc = args.rundir + "/schedules/"

    ## set up configparser to get original sim location
    config = configparser.ConfigParser()
    config.read(original_params)

    ## get info from params file
    obs_scenario_dir = config.get("params", "obs_scenario")
    assumed_mag = float(config.get("params", "KNe_mag_AB"))
    dist_measure = config.get("params", "distance_measure")

    ## additional vars for sims
    fitsloc = obs_scenario_dir + "/allsky/"
    injections = obs_scenario_dir + "/injections.dat"

    allsky_dir = obs_scenario_dir + "/allsky/"

    ## run the script
    compute_success(
        allsky_select,
        injections,
        fitsloc,
        schedloc,
        args.rundir,
        assumed_mag,
        dist_measure,
        prior_min=args.prior_min,
        prior_max=args.prior_max,
    )
