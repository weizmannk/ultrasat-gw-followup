#!/usr/bin/env python
# coding: utf-8

## This script contains some code to make various plots for the UVEX ToO follow-up of LVK observing scenarios, plus get some basic statistics of interest.

# python3 make-coverage-plots.py /path/to/params.ini

from pathlib import Path
from astropy.table import Table, join
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from matplotlib.ticker import MaxNLocator
import os, sys, configparser
import argparse
import scipy.stats as st
from scipy import special, integrate, optimize
import json


def get_trigger_quote(
    area_cut, astro_rate, sim_rate, run="O5", run_dur=1, event_type="bns"
):

    s2a_conversion_factor = get_s2a_with_err(*astro_rate, sim_rate, duration=run_dur)
    allsky = pd.read_csv(
        ".data/{}/{}_astro/allsky.dat".format(run, event_type), skiprows=1, sep="\t"
    )
    cut_idx = allsky["area(90)"] <= area_cut
    N = len(allsky[cut_idx]) * s2a_conversion_factor

    return N


@np.vectorize
def poisson_lognormal_rate_cdf(k, mu, sigma, duration=1):
    lognorm_pdf = st.lognorm(s=sigma, scale=np.exp(mu)).pdf

    def func(lam):
        prior = lognorm_pdf(lam)
        ## lam is rate * 1 yr; lam_adj is rate * duration
        lam_adj = lam * duration
        poisson_pdf = np.exp(
            special.xlogy(k, lam_adj) - special.gammaln(k + 1) - lam_adj
        )
        poisson_cdf = special.gammaincc(k + 1, lam_adj)
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
def poisson_lognormal_rate_quantiles(p, mu, sigma, duration=1):
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
        return poisson_lognormal_rate_cdf(k, mu, sigma, duration) - p

    if func(0) >= 0:
        return 0

    result = optimize.root_scalar(func, bracket=[0, 1e6])
    return result.root


def get_s2a_with_err(
    astro_rate_med, astro_rate_low, astro_rate_high, sim_rate, duration=1
):
    """
    Function to return median and 90% C.I. simulation-to-astrophysical conversion factor for a given duration and astrophysical rate median and 90% C.I.

    Expected median # of triggers = (astro_rate/sim_rate) * duration

    But 90% C.I. for # of triggers don't scale linearly with time, need to account for Poisson count error also

    We want conversion factor from N_sim (# of simulated events meeting our criteria) to N_astro (predicted # of astrophysical events meeting our critera).
    This will be N_astro = N_sim * (N_astro,tot)/(R_sim * 1yr * V_obs),
    where N_astro,tot is the expected total number of astrophysical events occuring for a given observation duration within the observed volume
    R_sim is the simulated rate per year per Gpc^3
    and N_astro,tot is the median and 90% C.I. of a Poisson distribution with a log-normal prior for R_astro with 90% C.I. (+X/-Y)

    Arguments
    -----------
    astro_rate_med (float) : The median astrophysical rate in yr^-1 Gpc^-3
    astro_rate_low (float) : The lower 90% C.I. for the astrophysical rate
    astro_rate_high (float): The upper 90% C.I. for the astrophysical rate
    sim_rate (float) : The simulated rate in yr^-1 Gpc^-3 (NOTE: given in yr^-1 Mpc^-3 by kiendrebeogo et al.2023, need to convert first!)
    duration (float) : Observation duration in years

    Returns
    -----------
    s2a_factors (array) : [med, low, high] conversion factors
    """

    ## get mu and sigma for the log-normal from the astrophysical median and 90% C.I.
    mu = np.log(astro_rate_med)
    sigma = (np.log(astro_rate_high) - np.log(astro_rate_low)) / np.diff(
        st.norm.interval(0.9)
    )

    ## get quantiles
    astro_count_med = astro_rate_med * duration
    astro_count_low, astro_count_high = poisson_lognormal_rate_quantiles(
        [0.05, 0.95], mu, sigma, duration=duration
    )

    s2a_med = astro_count_med / sim_rate
    s2a_low = astro_count_low / sim_rate
    s2a_high = astro_count_high / sim_rate

    return np.array([s2a_med, s2a_low, s2a_high])


def f2s(val):
    """
    Simple function to convert a float to a string with 1 decimal place.
    """
    return "{:0.1f}".format(val)


def arr2bounds(arr, fmt="default"):
    """
    Converts the output of get_s2a_with_err() into a string.

    Arguments
    -----------------------------
    arr (numpy array) : np.array([median, lower bound, upper bound])
    fmt (str)         : Format ('default' or latex').

    Returns
    -----------------------------
    Readable/LaTeX-parse-able string.
    """
    ## shouldn't allow for negative error bounds
    med = arr[0]
    high = arr[2] - arr[0]
    low = arr[0] - arr[1]
    if med - low < 0:
        low = med
    if fmt == "default":
        stat_string = f2s(med) + " (+" + f2s(high) + " / -" + f2s(low) + ")"
    elif fmt == "latex":
        stat_string = "$" + f2s(med) + "^{+" + f2s(high) + "}_{-" + f2s(low) + "}$"
    else:
        raise ValueError("Unknown string format. Can be 'default' or 'latex'.")

    return stat_string


def get_plots_and_stats(
    allsky_file,
    coverage_file,
    coverage_plots_dir,
    statistics_dir,
    outdir,
    N_batch,
    bandpass,
    absmag_mean,
    astro_rate,
    sim_rate,
    run_duration,
    max_texp,
    max_area,
    fov,
    coverage_threshold=90,
):
    """
    Function to compute follow-up statistics and make relevant plots.

    The statistics quoted account for the combination of lognormal astrophysical rate uncertainty and Poisson count error.

    Simulated event rate can be found from an observing scenario by doing:
    > sqlite3 events.sqlite
    > select comment from process;
    This will give you the simulated rate in yr^-1 Mpc^-3.

    Arguments
    --------------------------------
    allsky_file (str) : /path/to/allsky.dat
    coverage_file (str) : /path/to/coverage_file.csv (as produced by compute_tiling.py)
    outdir (str)        : /path/to/save/directory/
    N_batch (int)       : Number of batches used for preprocessing.
    bandpass (str)          : UV band being considered.
    absmag_mean (float)      : Assumed kilonova absolute bolometric magnitude.
    astro_rate (list of floats) : Astrophysical rate estimate to use. Must be given as [median, lower bound, upper bound] in yr^-1 Gpc^-3.
    sim_rate (float)            : Simulated event rate in yr^-1 Gpc^-3. See note above.
    run_duration (float)        : Duration of the observing run to be simulated, in years.
    max_texp (float)            : Maximum allowed per-field exposure time.
    coverage_threshold (float)  : Percent coverage of the GW 90% credible localization required to select an event as a ToO trigger. Default 90.

    """
    ## note that this varies from simulation to simulation
    ## find yours by doing:
    ## > sqlite3 events.sqlite
    ## > select comment from process;
    ## This will give you the simulated rate

    simrate_to_astrorate = get_s2a_with_err(
        *astro_rate, sim_rate, duration=run_duration
    )

    ## load events
    events_all = pd.read_csv(allsky_file)
    events_sched = pd.read_csv(coverage_file)

    ## filter to events that are well-covered and account for sun exclusion
    ## improvement point: could actually query where the sun is at observation time, define exclusion radius, and check for overlap with localization region.
    ## This could be a fun project for someone!
    # Extract event IDs and scheduled exposure times for events meeting the coverage threshold
    obs_id_list = events_sched.loc[
        events_sched["percent_coverage"] >= coverage_threshold, "event_id"
    ].to_list()
    obs_texp_list = events_sched.loc[
        events_sched["percent_coverage"] >= coverage_threshold, "texp_sched (ks)"
    ].to_list()

    # Identify events with less than 5% coverage
    low_coverage_ids = events_sched.loc[
        events_sched["percent_coverage"] < 5, "event_id"
    ].to_list()

    # Select events from events_all that have low coverage
    events_lowcov = events_all[events_all["simulation_id"].isin(low_coverage_ids)]

    # Filter events that are outside the Field of Regard (FoR)
    outside_FoR_id_list = events_lowcov[
        (events_lowcov["area(90)"] < max_area) & (events_lowcov["distmean"] < 300)
    ]["simulation_id"].to_list()

    events_texp = pd.read_csv(
        f"{outdir}/texp_out/allsky_texp_max_{bandpass}_batch0.csv"
    )
    if N_batch > 1:
        for i in range(1, N_batch):
            next_batch = pd.read_csv(
                f"{outdir}/texp_out/allsky_texp_max_{bandpass}_batch{i}.csv"
            )
            events_texp = pd.concat([events_texp, next_batch], ignore_index=True)

    texp_cut_id_list = events_texp.loc[
        events_texp["texp_max (s)"] > max_texp, "event_id"
    ].tolist()
    events_texp_reject = events_all[events_all["simulation_id"].isin(texp_cut_id_list)]
    events_texp_reject_id_list = events_texp_reject["simulation_id"].tolist()

    ## anything, regardless of other factors, that gets <1% coverage gets marked with black x
    ## mark out # of tiles = 0 FoR
    ## texp long and rejected gets marked with grey x
    ## three rejection categories: completely excluded, partially excluded, can't be tiled under time constraints

    # Identify events that are not fully covered (coverage < coverage_threshold)
    not_covered_event_ids = events_sched.loc[
        events_sched["percent_coverage"] < coverage_threshold, "event_id"
    ]
    events_notcov = events_all[events_all["simulation_id"].isin(not_covered_event_ids)]

    # Identify zero events with coverage < 0%
    zero_coverage_event_ids = events_sched.loc[
        events_sched["percent_coverage"] < 0.01, "event_id"
    ]
    events_0cov = events_notcov[
        events_notcov["simulation_id"].isin(zero_coverage_event_ids)
    ]
    events_0cov_id_list = events_0cov["simulation_id"].tolist()

    # Identify events with partial coverage (≥ 0% but < coverage_threshold)
    semi_covered_event_ids = events_sched.loc[
        events_sched["percent_coverage"] >= 0.01, "event_id"
    ]
    events_semicov = events_notcov[
        events_notcov["simulation_id"].isin(semi_covered_event_ids)
    ]
    events_semicov_sched = events_sched[
        events_sched["event_id"].isin(events_semicov["simulation_id"])
    ]

    ########
    ## see if event is in principle coverable under ideal circumstances
    coverable_filt = np.floor(
        (max_texp / 1e3) / events_semicov_sched["texp_sched (ks)"].to_numpy()
    ) >= np.ceil(
        events_semicov["area(90)"].to_numpy() / fov
    )  # True if coverable in perfect world
    events_coverable = events_semicov[coverable_filt]
    events_coverable_id_list = events_coverable["simulation_id"].to_list()
    events_timex = events_semicov[~coverable_filt]
    events_timex_id_list = events_timex["simulation_id"].to_list()

    ## get which events in the full simulation are observed
    ## True if observed, False if not
    obs_filter = events_all["simulation_id"].isin(obs_id_list)
    full_exclusion_filt = events_all["simulation_id"].isin(events_0cov_id_list)
    partial_exclusion_filt = events_all["simulation_id"].isin(events_coverable_id_list)
    time_exclusion_filt = events_all["simulation_id"].isin(events_timex_id_list)
    texp_reject_filt = events_all["simulation_id"].isin(events_texp_reject_id_list)

    # Combine filters
    combined_filter = (
        obs_filter | full_exclusion_filt | partial_exclusion_filt | time_exclusion_filt
    )

    ## get statistics on number of events *in princple* coverable vs. those we actually observe
    ## THESE SHOULD BE UPDATED LATER TO ACCOUNT FOR THE CATEGORIZATION ABOVE
    events_midcov = events_sched[
        (events_sched["percent_coverage"] < coverage_threshold)
    ].copy()
    midcov_areas = [
        events_all[events_all["simulation_id"] == ev_id]["area(90)"].to_numpy()[0]
        for ev_id in events_midcov["event_id"]
    ]
    events_midcov["area(90)"] = midcov_areas
    midcov_frac_coverable = len(
        events_midcov[
            np.round((max_texp / 1e3) / events_midcov["texp_sched (ks)"])
            > np.round(events_midcov["area(90)"] / fov)
        ]
    ) / len(events_midcov)
    midcov_ncoverable = midcov_frac_coverable * len(events_midcov)
    frac_obs_vs_coverable = np.sum(obs_filter) / (
        np.sum(obs_filter) + midcov_ncoverable
    )

    ## generate event lists for plot
    obs_dist = events_all[obs_filter]["distmean"].to_list()
    obs_area = events_all[obs_filter]["area(90)"].to_list()
    unobs_dist = events_all[np.invert(combined_filter)]["distmean"].to_list()
    unobs_area = events_all[np.invert(combined_filter)]["area(90)"].to_list()
    full_exclusion_dist = events_all[full_exclusion_filt]["distmean"].to_list()
    full_exclusion_area = events_all[full_exclusion_filt]["area(90)"].to_list()
    partial_exclusion_dist = events_all[partial_exclusion_filt]["distmean"].to_list()
    partial_exclusion_area = events_all[partial_exclusion_filt]["area(90)"].to_list()
    time_exclusion_dist = events_all[time_exclusion_filt]["distmean"].to_list()
    time_exclusion_area = events_all[time_exclusion_filt]["area(90)"].to_list()
    texp_reject_dist = events_all[texp_reject_filt]["distmean"].to_list()
    texp_reject_area = events_all[texp_reject_filt]["area(90)"].to_list()

    ## save all info for selected events
    selected_filter = events_sched["event_id"].isin(obs_id_list)
    events_selected_gwinfo = events_all[obs_filter].copy()
    events_selected_eminfo = events_sched[selected_filter].copy()
    events_selected_allinfo = pd.concat(
        [
            events_selected_gwinfo.reset_index(drop=True),
            events_selected_eminfo.reset_index(drop=True),
        ],
        axis=1,
    )
    # csv_savename = os.path.join(outdir,"allsky_selected_em.csv".format(bandpass,absmag_mean))
    csv_savename = os.path.join(
        statistics_dir, f"allsky_selected_em_{bandpass}_{abs(absmag_mean)}.csv"
    )
    events_selected_eminfo.to_csv(csv_savename, index=False)
    csv_savename_gwem = os.path.join(
        statistics_dir, "allsky_selected_gwem.csv".format(bandpass, absmag_mean)
    )
    events_selected_allinfo.to_csv(csv_savename_gwem, index=False)

    stat_savename = os.path.join(
        statistics_dir,
        f"ultrasat_event_statistics_{bandpass}_MAB_{abs(absmag_mean)}.txt",
    )

    print("Calculating statistics...")
    with open(stat_savename, "w") as outfile:
        print(
            "Number of selected events is",
            len(obs_dist),
            "; this is",
            100 * len(obs_dist) / len(events_all),
            "percent of the catalogue.",
            file=outfile,
        )
        ## covert to predictions for yearly rate
        print("Using a simulated rate of {:0.4E} yr^-1 Gpc^-3.".format(sim_rate))
        print(
            "Check that this is accurate for your simulations; see comments at beginning of this script for more details."
        )
        print(
            "The following predictions use a median (astrophysical rate / simulated rate) conversion factor of {:0.4f}.".format(
                simrate_to_astrorate[0]
            ),
            file=outfile,
        )
        print(
            "The uncertanty given includes the error contributions from lognormal astrophysical rate uncertainty and Poisson count statistics.",
            file=outfile,
        )
        print(
            "Total number of events is "
            + arr2bounds(arr=len(events_all) * simrate_to_astrorate, fmt="latex"),
            file=outfile,
        )
        print(
            "Predicted number of selected events (in {} yr) is ".format(run_duration)
            + arr2bounds(arr=len(obs_dist) * simrate_to_astrorate, fmt="latex"),
            file=outfile,
        )
        frac_obs_vs_coverable = np.sum(obs_filter) / (
            np.sum(obs_filter) + midcov_ncoverable
        )
        print(
            "Fraction of coverable events within ULTRSAT field of regard:",
            frac_obs_vs_coverable,
            file=outfile,
        )
        print(
            "(i.e., fraction of events lost to some level of sun exclusion is {:0.2f})".format(
                (1 - frac_obs_vs_coverable)
            ),
            file=outfile,
        )
        ## exposure time statistics
        obs_texp_arr = np.array(obs_texp_list) * 1000
        median_texp_sel = np.median(obs_texp_arr)
        min_texp_sel = np.min(obs_texp_arr)
        max_texp_sel = np.max(obs_texp_arr)
        print("Exposure time statistics for selected events (in s):", file=outfile)
        print(
            "Median:",
            median_texp_sel,
            "; Min:",
            min_texp_sel,
            "; Max:",
            max_texp_sel,
            file=outfile,
        )
        ## tiling statistics
        obs_tile_list = events_sched[
            [(ev_id in obs_id_list) for ev_id in events_sched["event_id"]]
        ]["tiles_to_90pct"].to_list()
        obs_tile_arr = np.array(obs_tile_list)
        median_tile_sel = np.median(obs_tile_arr)
        min_tile_sel = np.min(obs_tile_arr)
        max_tile_sel = np.max(obs_tile_arr)
        print(
            "Tiling statistics for selected events (tiles to 90% coverage of 90% localization):",
            file=outfile,
        )
        print(
            "Median:",
            median_tile_sel,
            "; Min:",
            min_tile_sel,
            "; Max:",
            max_tile_sel,
            file=outfile,
        )
        frac = np.sum(obs_tile_arr <= 5) / len(obs_tile_arr)
        print(
            "Fraction of selected events covered in <=5 tiles is ", frac, file=outfile
        )
        print(
            "This corresponds to {} predicted events in {} yr (90% C.I.).".format(
                arr2bounds(
                    arr=frac * len(obs_dist) * simrate_to_astrorate, fmt="latex"
                ),
                run_duration,
            ),
            file=outfile,
        )
    print("Statistics saved to {}.".format(stat_savename))

    print("Making plots...")
    with plt.style.context("seaborn-v0_8-talk"):
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        #     ax.set_yscale('log')
        ax.set_xlim(1e2, 1e3)
        ax.set_ylim(1, 1e4)
        ax.grid(which="both", axis="x")
        ax.grid()
        # ax.plot(O5_events['distmean'], O5_events['area(90)'], '.', ms=6, label='O5 events')
        ## unselected events
        plt.scatter(
            unobs_dist,
            unobs_area,
            marker=".",
            s=8,
            label="Unselected events",
            color="darkgray",
        )
        ## selected but unobserved by category
        plt.scatter(
            full_exclusion_dist + partial_exclusion_dist,
            full_exclusion_area + partial_exclusion_area,
            marker="x",
            s=14,
            linewidth=0.75,
            label="Complete or partial field of regard exclusion",
            color="black",
        )
        #     plt.scatter(partial_exclusion_dist, partial_exclusion_area, marker='x', s=10,
        #             label='Partial field of regard exclusion', color='slategrey')
        plt.scatter(
            time_exclusion_dist,
            time_exclusion_area,
            marker="x",
            s=15,
            linewidth=0.75,
            label="Total epoch time constraint",
            color="darkorange",
        )
        plt.scatter(
            texp_reject_dist,
            texp_reject_area,
            marker="x",
            s=15,
            linewidth=0.75,
            label="__nolabel__",
            color="darkorange",
        )
        ##observed
        plt.scatter(
            obs_dist,
            obs_area,
            marker="o",
            s=17,
            linewidth=1,
            label="Selected events",
            c=np.array(obs_texp_list) * 1000,
            cmap=cm.cool_r,
        )

        # ax.fill_between(DL, A, ax1.get_ylim()[1], color='lightgray')
        # ax.plot(DL, A, color='gray', lw=4)

        ax.set_xlabel("Luminosity distance (Mpc)")
        ax.set_ylabel("90% credible area (deg$^2$)")
        cbar = plt.colorbar()
        cbar.set_label("Exposure Time (s)")  # , rotation=270)
        plt.legend(loc="upper left")

        plt.title(
            f"Event Selection for ULTRASAT ToO with {bandpass.upper()} $M_{{AB}}=${absmag_mean:0.1f}"
        )
        plot_savename = os.path.join(
            coverage_plots_dir,
            f"ultrasat_event_selection_{bandpass}_MAB_{abs(absmag_mean)}.png",
        )
        plt.savefig(plot_savename, bbox_inches="tight", dpi=300)
        plot_savename_pdf = os.path.join(
            coverage_plots_dir,
            f"ultrasat_event_selection_{bandpass}_MAB_{abs(absmag_mean)}.pdf",
        )
        plt.savefig(plot_savename_pdf, bbox_inches="tight", dpi=300)
        plt.close()

    plt.figure()
    plt.hist(obs_texp_arr, bins=20, color="mediumorchid")
    plt.title("Histogram of Exposure Times")
    plt.xlabel("Exposure Time (s)")
    plt.ylabel("Count")
    hist1_savename = os.path.join(
        coverage_plots_dir,
        f"ultrasat_texp_histogram_{bandpass}_MAB_{abs(absmag_mean)}.png",
    )
    plt.savefig(hist1_savename, bbox_inches="tight")
    hist1_savename_pdf = os.path.join(
        coverage_plots_dir,
        f"ultrasat_texp_histogram_{bandpass}_MAB_{abs(absmag_mean)}.pdf",
    )
    plt.savefig(hist1_savename_pdf, bbox_inches="tight")
    plt.close()

    plt.figure()
    ax = plt.gca()
    plt.hist(obs_tile_arr, bins=20, color="mediumorchid")
    plt.title(
        f"Histogram of Tiles to Reach {coverage_threshold}% Coverage of 90% c.l. Localization Region"
    )
    plt.xlabel("Tiles")
    plt.ylabel("Count")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    hist2_savename = os.path.join(
        coverage_plots_dir,
        f"ultrasat_tiling_histogram_{bandpass}_MAB_{abs(absmag_mean)}.png",
    )
    plt.savefig(hist2_savename, bbox_inches="tight")
    hist2_savename_pdf = os.path.join(
        coverage_plots_dir,
        f"ultrasat_tiling_histogram_{bandpass}_MAB_{abs(absmag_mean)}.pdf",
    )
    plt.savefig(hist2_savename_pdf, bbox_inches="tight")
    plt.close()

    return


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Given observing schedules, compute tiling and statistics."
    )
    parser.add_argument("params", type=str, help="/path/to/params_file.ini")

    args = parser.parse_args()

    # Set up config parser
    config = configparser.ConfigParser()
    config.read(args.params)

    # Get info from params file
    # obs_scenario_dir   = config.get("params", "obs_scenario")
    outdir = config.get("params", "save_directory")
    N_batch = config.getint("params", "N_batch_preproc")
    bandpass = config.get("params", "bandpass")
    absmag_mean = config.getfloat("params", "absmag_mean")

    astro_bns_median = config.getfloat("params", "astro_bns_median")
    astro_nsbh_median = config.getfloat("params", "astro_nsbh_median")
    astro_bns_interval_90 = json.loads(config.get("params", "astro_bns_interval_90"))
    astro_rate_bounds = json.loads(config.get("params", "astro_nsbh_interval_90"))
    sim_rate = config.getfloat("params", "sim_bns_rate")

    run_duration = config.getfloat("params", "obs_duration")
    max_texp = config.getfloat("params", "max_texp")
    coverage_threshold = config.getfloat("params", "coverage_threshold", fallback=90)
    max_area = config.getfloat("params", "max_area")
    fov = config.getfloat("params", "fov")

    BNS = config.getboolean("params", "BNS", fallback=True)
    NSBH = config.getboolean("params", "NSBH", fallback=True)
    BBH = config.getboolean("params", "BBH", fallback=False)

    # Define paths
    coverage_plots_dir = os.path.join(outdir, "coverage_plots")
    statistics_dir = os.path.join(outdir, "statistics")

    # Ensure directories exist
    os.makedirs(coverage_plots_dir, exist_ok=True)
    os.makedirs(statistics_dir, exist_ok=True)

    # Filtering selected populations
    populations = [
        pop_name
        for pop_name, pop_value in zip(["BNS", "NSBH", "BBH"], [BNS, NSBH, BBH])
        if pop_value
    ]
    allsky_filename = f"allsky_{'_'.join(populations).lower()}"

    if populations == ["NSBH"]:
        astro_rate_median = astro_nsbh_median
        astro_rate_bounds = astro_nsbh_interval_90
    else:
        astro_rate_median = astro_bns_median
        astro_rate_bounds = astro_bns_interval_90

    allsky_file = os.path.join(outdir, f"{allsky_filename}.csv")
    coverage_file = os.path.join(outdir, "allsky_coverage.csv")

    astro_rate = [astro_rate_median] + astro_rate_bounds

    get_plots_and_stats(
        allsky_file,
        coverage_file,
        coverage_plots_dir,
        statistics_dir,
        outdir,
        N_batch,
        bandpass,
        absmag_mean,
        astro_rate,
        sim_rate,
        run_duration,
        max_texp,
        max_area,
        fov,
        coverage_threshold,
    )
