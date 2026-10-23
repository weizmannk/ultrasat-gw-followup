"""Follow-up efficiency in the localization-area / luminosity-distance plane.

One figure per observing run: selected events as stars whose size encodes the
scheduler objective and whose colour encodes the detection probability,
missing events as grey circles, marginal histograms of distance and area, and
the boundary of what the mission can reach at all (maximum area, maximum
distance, and the ``area ~ d^-4`` segment between them).

Usage
-----
::

    from ultrasat_gw_followup.analysis.area_distance import plot_area_distance

    plot_area_distance("results/ultrasat_lcs.ecsv", outdir="paper_outputs")

Outputs ``<mission>_<skygrid>_area-distance_<run>.pdf`` in ``outdir``.
"""

import logging
import warnings

import numpy as np
import synphot
from astropy import units as u
from astropy.coordinates import ICRS
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy.table import QTable
from astropy.time import Time
from astropy.visualization import quantity_support
from astropy_healpix import HEALPix
from m4opt import missions
from m4opt.synphot import observing
from matplotlib import colors as mcolors
from matplotlib import gridspec, patheffects
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from tqdm.auto import tqdm

from ultrasat_gw_followup.analysis import get_skygrid


def fov_area(mission):
    """Instantaneous field-of-view area of a mission, analytically.

    ``m4opt-paper`` computes this as ``fov.width * fov.height``, which assumes
    a single rectangular footprint. That holds for ULTRASAT and is used
    unchanged here. UVEX, however, exposes ``mission.fov`` as a ``Regions``
    collection of nine detector polygons with no ``width``, so the areas of
    the tiles are summed instead.

    The tiles are summed rather than unioned because they do not overlap:
    the sum, 12.3604 deg^2, agrees with the HEALPix rasterisation of the same
    footprint to 0.02% once that is converged (12.3628 deg^2 at nside=8192).
    Rasterising is what this function used to do, and at a usable resolution
    it overestimated by about 1%, since boundary pixels count whole.

    Each polygon is projected on the tangent plane at its own declination and
    closed with the shoelace formula. Over a field a few degrees across the
    curvature correction is of order 1e-4 and negligible here.

    Note that the published UVEX figure of 12.25 deg^2 is the nominal
    3.5 x 3.5 deg specification, not the active area of the nine chips.

    Parameters
    ----------
    mission : m4opt.missions.Mission
        Mission whose ``fov`` is read.

    Returns
    -------
    astropy.units.Quantity
        Field-of-view area, in square degrees.
    """
    region = mission.fov
    width = getattr(region, "width", None)
    if width is not None:
        return (width * region.height).to(u.deg**2)

    area = 0.0
    for polygon in region:
        vertices = polygon.vertices
        # Tangent plane: compress right ascension by cos(dec) so that the
        # shoelace formula measures a solid angle rather than a coordinate
        # area.
        x = ((vertices.ra.deg + 180) % 360 - 180) * np.cos(np.radians(vertices.dec.deg))
        y = vertices.dec.deg
        area += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area * u.deg**2


# Suppress known warnings from astropy and lal
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
warnings.filterwarnings("ignore", ".*dubious year.*")
warnings.filterwarnings(
    "ignore", "Tried to get polar motions for times after IERS data is valid.*"
)

log = logging.getLogger(__name__)


def customize_style():
    """Apply the figure style of the paper."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11


# ---------------------------------------------------------------------------
# Theoretical limiting magnitude
# ---------------------------------------------------------------------------
def compute_theoretical_limmag(
    mission,
    hpx: HEALPix,
    constants: dict,
    date: Time | None = None,
    time_step: u.Quantity = 1 * u.hour,
) -> u.Quantity:
    """
    Compute the theoretical best-case limiting magnitude over the full sky.

    For ULTRASAT (GEO orbit), the Cerenkov background varies along the orbit
    as the satellite traverses different regions of the Van Allen radiation
    belts (AE8 model). To capture this variation, the orbit is propagated
    over one full sidereal day (one GEO period ≈ 24 h) and limmag is
    evaluated at each orbital position over all HEALPix pixels.

    Following the same pattern as M4OPT's scheduler, all orbital positions
    are computed at once via ``mission.observer_location(obstimes)`` to
    ensure type compatibility with the AE8 Cerenkov model.

    Non-observable pixels (blocked by Earth limb, Sun, or Moon constraints)
    return ``-inf`` and are excluded from the maximum.

    Parameters
    ----------
    mission : m4opt Mission
        Mission object (e.g. ``ultrasat``, ``uvex``).
    hpx : HEALPix
        HEALPix grid used for full-sky evaluation.
    constants : dict
        Scheduling constants from the plan metadata.
        Required keys: ``snr``, ``deadline``, ``delay``,
        ``exptime_max``, ``bandpass``.
    date : astropy.time.Time, optional
        Reference date for orbit propagation. Defaults to 2026-09-01.
    time_step : astropy.units.Quantity, optional
        Time step between orbital samples (default: 1 hour).
        For GEO orbits (ULTRASAT), 1 hour is sufficient since the Cerenkov
        background varies slowly. Use smaller steps for LEO missions.

    Returns
    -------
    astropy.units.Quantity
        Best-case limiting magnitude in AB mag, maximised over all
        observable sky pixels and all sampled orbital positions.
    """
    if date is None:
        date = Time("2026-09-01")

    # Propagate orbit over one full GEO period (24 h) — same pattern as M4OPT
    obstimes = date + np.arange(0, 24 * u.hour, time_step, like=time_step)
    observer_locations = mission.observer_location(obstimes)

    exptime = min(
        constants["deadline"] - constants["delay"],
        constants["exptime_max"],
    )
    flat_spectrum = synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag)
    all_pixels = np.arange(hpx.npix)

    def _limmag_at_time(obs_location, obs_time: Time) -> float:
        """Evaluate limmag over all sky pixels at a single orbital position.

        Returns the maximum over finite pixels, or ``-inf`` if no pixel
        is observable at this time.
        """
        with observing(
            observer_location=obs_location,
            target_coord=hpx.healpix_to_skycoord(all_pixels),
            obstime=obs_time,
        ):
            # Cerenkov / AE8 background (ULTRASAT only) is part of the mission
            # model in m4opt >= 2.13 and is evaluated from this observing()
            # context; missions without it (e.g. UVEX) are unaffected.
            limmag = mission.detector.get_limmag(
                constants["snr"],
                exptime,
                flat_spectrum,
                constants["bandpass"],
            )

        # Exclude non-observable pixels (-inf from Earth limb, Sun, Moon)
        finite = np.isfinite(limmag.to_value(u.mag))
        if not np.any(finite):
            log.warning(f"No observable pixels at {obs_time.iso} — skipping.")
            return -np.inf

        log.debug(
            f"  {obs_time.iso} = --> "
            f"limmag [{limmag[finite].min():.3f}, {limmag[finite].max():.3f}] "
            f"({finite.sum()}/{hpx.npix} observable pixels)"
        )
        return limmag[finite].max().to_value(u.mag)

    limmag_best = max(
        _limmag_at_time(obs_loc, obs_time)
        for obs_loc, obs_time in tqdm(
            zip(observer_locations, obstimes),
            desc=f"limmag ({mission.name})",
            total=len(obstimes),
            unit="step",
        )
    )

    log.info(
        f"Theoretical limmag {mission.name} = {limmag_best:.3f} mag "
        f"({len(obstimes)} steps x {time_step}/step x {hpx.npix} pixels)"
    )
    return limmag_best * u.mag


def plot_area_distance(events_file, outdir="output", show=False):
    """Draw one area-distance figure per observing run of a result table.

    An event is triggered when its objective reaches the cutoff stored in the
    table, and missing otherwise. The detection histograms are weighted by the
    detection probability, and a source-frame mass cut at 3 solar masses is
    applied before plotting.

    Parameters
    ----------
    events_file : str or pathlib.Path
        Post-processed result table. Read for ``distance``, ``area(90)``,
        ``objective_value``, ``detection_probability_known_position``,
        ``mass1``, ``mass2``, ``run``, and the scheduler settings stored in
        every row.
    outdir : str or pathlib.Path, optional
        Directory the figures are written to.
    show : bool, optional
        Display each figure instead of closing it.

    Raises
    ------
    ValueError
        If a setting that must be constant over the table is not.
    """
    # Enable astropy units in matplotlib
    quantity_support()
    customize_style()

    # Load event data from ECSV file
    main_table = QTable.read(events_file)

    # ===================================================================
    # Extract constant parameters from table
    # ===================================================================
    # These values should be identical across all rows
    constant_cols = [
        "cutoff",
        "mission",
        "nside",
        "snr",
        "deadline",
        "delay",
        "exptime_min",
        "exptime_max",
        "bandpass",
        "absmag_mean",
        "absmag_stdev",
        "visits",
        "skydir",
    ]

    constants = {}
    for col in constant_cols:
        if col in main_table.columns:
            # Verify that column contains a single unique value
            if not np.all(main_table[col] == main_table[0][col]):
                raise ValueError(
                    f"Column '{col}' has multiple values; expected a constant."
                )
            constants[col] = main_table[0][col]

    # ===================================================================
    # Setup observational parameters
    # ===================================================================
    mission = getattr(missions, constants["mission"])
    cutoff = constants["cutoff"]  # Threshold for triggered vs missing

    #  This Function need to be run with Cplex credendials , so its hardcoded to ovoid rerunning it every time
    hpx = HEALPix(nside=constants["nside"], frame=ICRS(), order="nested")

    log.info(mission.name)
    if mission.name == "ultrasat":
        limmag = 23.576296607302744 * u.mag  # Limiting magnitude
        # ULTRASATneed cplex to propagate around the orbit

    # elif  mission.name == "uvex":
    #     with observing(
    #         observer_location=EarthLocation(0 * u.m, 0 * u.m, 0 * u.m),
    #         target_coord=hpx.healpix_to_skycoord(np.arange(hpx.npix)),
    #         obstime=Time("2026-05-14"),
    #     ):
    #         limmag = mission.detector.get_limmag(
    #             constants["snr"],
    #             min(constants["deadline"] - constants["delay"],
    #                 constants["exptime_max"]),
    #             synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag),
    #             constants["bandpass"],
    #         ).max()
    #         log.info(f"limmag:{limmag}")
    else:
        limmag = compute_theoretical_limmag(
            mission,
            hpx,
            constants,
            date=Time("2026-05-14"),
            time_step=0.5 * u.h,
        )

    # ===================================================================
    # Calculate theoretical detection limits
    # ===================================================================
    # These define the boundaries of detectability in the area-distance plane

    min_area = fov_area(mission)

    # Full sky area in square degrees
    full_sky_area = (1 * u.spat).to(u.deg**2)  # 41,253 deg^2

    skymap_area_cl = 90  # Credible level for sky localization area
    # Maximum coverable area within the "deadline" time
    chisq_ppf = stats.chi2(df=2).ppf
    area_factor = chisq_ppf(skymap_area_cl / 100) / chisq_ppf(constants["cutoff"])
    max_area = (
        area_factor
        * min_area
        * (constants["deadline"] - constants["delay"])
        / (constants["visits"] * constants["exptime_min"])
    ).to(u.deg**2)

    # ratio et max_area_display
    ratio = max_area / full_sky_area

    if ratio > 1:
        max_area_display = full_sky_area  # float
        log.info(
            f"max_area ({max_area:.0f} deg2) > full_sky ({full_sky_area:.0f} deg2): "
            f"ratio = {ratio:.2f}x — using full sky as display limit"
        )
    else:
        max_area_display = max_area  # float
        log.info(
            f"max_area ({max_area:.0f} deg2) < full_sky — "
            f"using max_area as display limit\n"
            f"ratio = {ratio:.2f} x full sky area"
        )

    # Maximum distance: beyond this, fewer than `cutoff` of kilonovae would be
    # bright enough, so no plan can reach the objective threshold.
    #
    # Inverting the distance modulus, m - M = 5 log10(d / 10 pc):
    #
    #     d / Mpc = 10 ** (0.2 * (m_lim - M - 25))
    #
    # M is not the mean absolute magnitude but its bright tail,
    # M = absmag_mean - Phi^-1(1 - cutoff) * absmag_stdev. With cutoff = 0.1
    # that is the 10th percentile, M = -16 - 1.28 * 1.3 = -17.67: the distance
    # at which exactly `cutoff` of the kilonova population still clears the
    # limit, which is the same threshold the scheduler applies to P_obj.
    #
    # m_lim is the best case: the whole window on a single field, at the most
    # favourable sky position and orbital phase (see compute_theoretical_limmag).
    # The boundary is therefore a ceiling no plan can beat, not a prediction.
    max_distance = (
        10
        ** (
            0.2
            * (
                limmag.to_value(u.mag)
                - (
                    constants["absmag_mean"]
                    - stats.norm.ppf(1 - constants["cutoff"])
                    * constants["absmag_stdev"]
                )
                - 25
            )
        )
        * u.Mpc
    )

    # Crossover distance: where area constraint transitions from full sky to FOV-limited
    crossover_distance = (
        max_distance
        * (min_area / max_area_display).to_value(u.dimensionless_unscaled) ** 0.25
    )

    # ===================================================================
    # Process each observing run separately
    # ===================================================================

    # Extract mission skygrid from metadata
    skygrid = get_skygrid(main_table)

    runs = np.unique(main_table["run"])

    for run in runs:
        log.info(f"Processing run: {run}")

        # Filter events for current run
        table = main_table[main_table["run"] == run]
        # selected_table = table[table["objective_value"] >= cutoff]

        # ===================================================================
        # Apply source-frame mass filter
        # ===================================================================
        # Convert to source frame and filter out massive neutron stars
        z = np.array(
            [
                z_at_value(cosmo.luminosity_distance, d * u.Mpc).to_value(
                    u.dimensionless_unscaled
                )
                for d in table["distance"]
            ]
        )
        m2 = table["mass2"] / (1 + z)  # Convert detector-frame to source-frame mass
        table = table[m2 <= 3]  # Keep only neutron stars (m2 <= 3 solar masses)

        # ===================================================================
        # Classify events: Missing vs Triggered
        # ===================================================================
        # Missing: objective_value < cutoff (not observed)
        # Triggered: objective_value >= cutoff (observed)
        triggered_mask = table["objective_value"] >= cutoff
        not_triggered_table = table[~triggered_mask]
        triggered_table = table[triggered_mask]

        # ===================================================================
        # Define plot limits
        # ===================================================================
        xlim_min, xlim_max = 40, 4000  # Distance range in Mpc
        crossover_val = crossover_distance.to_value(u.Mpc)
        max_dist_val = max_distance.to_value(u.Mpc)
        # The ceiling of the boundary is whichever is smaller: the sky, or
        # what the mission can actually tile in the window. Hardcoding the
        # full sky here drew UVEX as if it could cover any localization,
        # and contradicted the crossover distance, which already uses
        # max_area_display.
        max_area_display_val = max_area_display.to_value(u.deg**2)
        min_area_val = min_area.to_value(u.deg**2)

        # Dynamic y-axis limit based on actual data
        if len(table) > 0:
            actual_max_area = table[f"area({skymap_area_cl})"].max()
        else:
            actual_max_area = 1000

        ylim_max = max(actual_max_area * 3.5, max_area_display_val * 1.3, 1000)
        ylim_min = 0.1

        # ===================================================================
        # Create figure with gridspec layout
        # ===================================================================
        # Layout: 4x4 grid with joint plot + marginal histograms
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(
            4,
            4,
            figure=fig,
            left=0.10,
            right=0.96,
            bottom=0.10,
            top=0.96,
            hspace=0.05,
            wspace=0.05,
        )

        ax_joint = fig.add_subplot(gs[1:4, 0:3])  # Main scatter plot
        ax_top = fig.add_subplot(
            gs[0, 0:3], sharex=ax_joint
        )  # Top histogram (distance)
        ax_right = fig.add_subplot(
            gs[1:4, 3], sharey=ax_joint
        )  # Right histogram (area)

        # Hide overlapping tick labels
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)

        # ===================================================================
        # Configure main plot axes
        # ===================================================================
        ax_joint.set_xlim(xlim_min, xlim_max)
        ax_joint.set_ylim(ylim_min, ylim_max)
        ax_joint.set_xscale("log")
        ax_joint.set_yscale("log")
        ax_joint.set_xlabel(
            r"Luminosity distance, $d_\mathrm{L}$ (Mpc)", fontsize=15, weight="bold"
        )
        ax_joint.set_ylabel(
            rf"{skymap_area_cl}% credible area (deg$^2$)", fontsize=15, weight="bold"
        )
        ax_joint.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color="gray")

        # ===================================================================
        # Draw theoretical limit boundaries
        # ===================================================================
        # Gray shaded region: undetectable parameter space
        ax_joint.fill_between(
            [xlim_min, crossover_val, max_dist_val, max_dist_val, xlim_max],
            [
                max_area_display_val,
                max_area_display_val,
                min_area_val,
                ylim_min,
                ylim_min,
            ],
            [ylim_max] * 5,
            color="lightgray",
            alpha=0.4,
            zorder=0,
        )

        # Black dashed line: detection boundary
        ax_joint.plot(
            [xlim_min, crossover_val, max_dist_val, max_dist_val],
            [max_area_display_val, max_area_display_val, min_area_val, ylim_min],
            color="black",
            linewidth=3,
            zorder=1,
            linestyle="--",
            alpha=0.8,
        )

        # ===================================================================
        # Add text annotations for theoretical limits
        # ===================================================================
        text_kwargs = {
            "color": "black",
            "fontsize": 11,
            "ha": "center",
            "va": "bottom",
            "path_effects": [patheffects.withStroke(linewidth=4, foreground="white")],
            "zorder": 10,
            "weight": "bold",
        }

        # "Max area" label on horizontal segment
        ax_joint.text(
            np.sqrt(xlim_min * crossover_val),
            max_area_display_val * 1.15,
            "Max area",
            **text_kwargs,
        )

        # "Max distance" label on vertical segment
        text_kwargs_vertical = text_kwargs.copy()
        text_kwargs_vertical.update({"va": "top", "ha": "center"})
        ax_joint.text(
            max_dist_val * 1.2,
            np.sqrt(ylim_min * min_area_val),
            "Max\ndistance",
            rotation=-90,
            **text_kwargs_vertical,
        )

        # "Area & d^{-4}" label on diagonal segment
        ax_joint.text(
            np.sqrt(crossover_val * max_dist_val),
            np.sqrt(min_area_val * max_area_display_val),
            r"Area $\propto d^{-4}$",
            rotation=-45,
            **text_kwargs,
        )

        # ===================================================================
        # Setup colormap for detection probability
        # ===================================================================
        # Rainbow: violet  -->blue -->cyan -->green -->yellow -->orange -->red (0  --> 1)
        cmap = plt.cm.rainbow
        norm = mcolors.Normalize(vmin=0, vmax=1)

        # ===================================================================
        # Scatter plot: Missing events (nearly invisible)
        # ===================================================================
        # Small gray circles with very low opacity
        # These form a subtle background showing all events
        ax_joint.scatter(
            not_triggered_table["distance"],
            not_triggered_table[f"area({skymap_area_cl})"],
            s=8,  # Small size
            marker="o",
            c="lightgray",
            edgecolor="none",
            alpha=0.8,  # Nearly transparent
            zorder=2,
            label="Missing",
        )

        # ===================================================================
        # Scatter plot: Triggered events (prominent stars)
        # ===================================================================
        # Large stars with dual encoding:
        # - Size & objective_value (how important to observe)
        # - Color & detection_probability (likelihood of detection)
        ax_joint.scatter(
            triggered_table["distance"],
            triggered_table[f"area({skymap_area_cl})"],
            s=triggered_table["objective_value"] * 400,  # Large stars (2x baseline)
            marker="*",
            c=triggered_table["detection_probability_known_position"],
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.2,  # Thin border for filled appearance
            alpha=1.0,  # Fully opaque
            zorder=10,
            label="Triggered",
        )

        # ===================================================================
        # Create 5x5 legend grid
        # ===================================================================
        # Shows all combinations of objective value (x) x detection prob (y)
        # Bounds in axes fraction, the geometry the mpl_toolkits inset locator
        # used to produce. The locator is avoided because reusing it under
        # savefig(bbox_inches="tight") raises on matplotlib >= 3.11.
        ax_legend = ax_joint.inset_axes((0.54, 0.05, 0.16, 0.16))

        ax_legend.margins(0.2)

        # Create 5x5 grid (0, 0.25, 0.5, 0.75, 1.0)
        dx = 1 / 4
        x = np.arange(0, 1 + dx, dx)
        ax_legend.set_xticks([0, 0.5, 1])
        ax_legend.set_yticks([0, 0.5, 1])
        ax_legend.set_xticklabels(["0.0", "0.5", "1.0"], fontsize=7)
        ax_legend.set_yticklabels(["0.0", "0.5", "1.0"], fontsize=7)
        ax_legend.tick_params(labelsize=7, length=2, width=0.8)

        # Create mesh grid for all combinations
        x_grid, y_grid = (a.ravel() for a in np.meshgrid(x, x))

        # Background: small gray circles at each grid point
        ax_legend.scatter(
            x_grid,
            y_grid,
            s=8,
            facecolor="silver",
            edgecolor="none",
            clip_on=False,
            zorder=1,
        )

        # Foreground: colored stars showing size x color encoding
        marker_scale = 200
        ax_legend.scatter(
            x_grid,
            y_grid,
            s=marker_scale * x_grid,  # Size & x (objective value)
            c=y_grid,  # Color & y (detection probability)
            norm=norm,
            cmap=cmap,
            marker="*",
            clip_on=False,
            edgecolor="black",
            linewidth=0.5,
            alpha=1.0,
            zorder=2,
        )

        # Add axis labels for the grid
        twin = ax_legend.twiny()
        twin.set_frame_on(False)
        twin.set_ylim(ax_legend.get_ylim())
        twin.set_xticks([])
        twin.set_xlabel("Objective val.", fontsize=8, weight="bold", labelpad=2)

        ax_legend.set_ylabel("Detection prob.", fontsize=8, weight="bold", labelpad=2)

        # ===================================================================
        # Style the legend grid box (Seaborn elegant style)
        # ===================================================================
        # Close the box on all four sides, in one uniform soft gray. The twiny
        # axes above only carries the "Objective val." label; its own frame is
        # off, so it contributes no competing black spine.
        for spine in ax_legend.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("gray")

        # White semi-transparent background so the grid reads over the scatter
        ax_legend.patch.set_facecolor("white")
        ax_legend.patch.set_alpha(0.95)
        ax_legend.patch.set_visible(True)

        # ===================================================================
        # Create text legend (Missing / Triggered)
        # ===================================================================

        # Define legend symbols
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Missing",
                markersize=5,
                markerfacecolor="black",
                markeredgecolor="gray",
                markeredgewidth=0.3,
                alpha=0.3,
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                label="Triggered",
                markersize=18,
                markerfacecolor="black",
                markeredgecolor="black",
                markeredgewidth=1,
            ),
        ]

        # Create legend with Seaborn elegant styling
        legend = ax_joint.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0, 0.93),  # Slightly lowered position
            fontsize=11,
            framealpha=0.9,  # Semi-transparent background
            edgecolor="#CCCCCC",  # Soft gray border
            fancybox=True,  # Rounded corners
            shadow=True,  # Subtle shadow for depth
            borderpad=0.8,  # Internal padding
            labelspacing=0.4,  # Spacing between items
            handletextpad=0.6,  # Space between symbol and text
            frameon=True,
            title_fontsize=10,
        )
        legend.get_frame().set_linewidth(1.2)  # Thin border
        legend.get_frame().set_facecolor("white")  # White background

        # ===================================================================
        # Create marginal histograms
        # ===================================================================
        # Bins in log space for proper visualization
        bins_dist = np.logspace(np.log10(xlim_min), np.log10(xlim_max), 30)
        bins_area = np.logspace(np.log10(ylim_min), np.log10(ylim_max), 30)

        # Detection probability used as weights for "Detected" category
        weights_detected = triggered_table["detection_probability_known_position"]

        # ===================================================================
        # Calculate 90th percentiles
        # ===================================================================
        perc_90_all = (
            np.quantile(table["distance"], [0.9], method="inverted_cdf").item()
            if len(table) > 0
            else 0
        )

        perc_90_trig = (
            np.quantile(
                triggered_table["distance"], [0.9], method="inverted_cdf"
            ).item()
            if len(triggered_table) > 0
            else 0
        )

        perc_90_det = (
            np.quantile(
                triggered_table["distance"],
                [0.9],
                weights=triggered_table["detection_probability_known_position"],
                method="inverted_cdf",
            ).item()
            if len(triggered_table) > 0
            else 0
        )

        # Calculate percentiles
        ticks = [
            np.quantile(table["distance"], [0.9], method="inverted_cdf").item(),
            np.quantile(
                triggered_table["distance"], [0.9], method="inverted_cdf"
            ).item(),
            np.quantile(
                triggered_table["distance"],
                [0.9],
                weights=triggered_table["detection_probability_known_position"],
                method="inverted_cdf",
            ).item(),
        ]
        colors_lines = ["gray", "dodgerblue", "red"]
        # Draw vertical lines
        for tick, color in zip(ticks, colors_lines):
            ax_top.axvline(tick, linewidth=2, color=color, alpha=0.9, zorder=4)

        # ===================================================================
        # Top histogram: Distance distribution
        # ===================================================================
        # Three overlaid histograms: All events, Triggered, Detected (weighted)
        ax_top.hist(
            table["distance"],
            bins=bins_dist,
            color="lightgray",
            alpha=0.6,
            label="All events",
            edgecolor="gray",
            linewidth=1,
            zorder=1,
        )
        ax_top.hist(
            triggered_table["distance"],
            bins=bins_dist,
            color="dodgerblue",
            alpha=0.7,
            label="Triggered",
            edgecolor="darkblue",
            linewidth=1,
            zorder=2,
        )
        ax_top.hist(
            triggered_table["distance"],
            bins=bins_dist,
            weights=weights_detected,
            color="red",
            alpha=0.8,
            label="Detected",
            edgecolor="darkred",
            linewidth=1.5,
            zorder=3,
        )

        # Configure axes
        ax_top.set_ylabel("Counts", fontsize=12, weight="bold")
        ax_top.set_xscale("log")
        ax_top.legend(
            loc="upper right",
            fontsize=10,
            framealpha=0.95,
            edgecolor="black",
            frameon=True,
            fancybox=False,
            ncol=3,
        )
        ax_top.grid(True, alpha=0.3, linestyle="-", color="gray")
        ax_top.set_ylim(bottom=0)

        # ===================================================================
        # Right histogram: Area distribution WITH 90th percentile lines
        # ===================================================================
        # Calculate area percentiles
        perc_90_area_all = (
            np.quantile(
                table[f"area({skymap_area_cl})"], [0.9], method="inverted_cdf"
            ).item()
            if len(table) > 0
            else 0
        )

        perc_90_area_trig = (
            np.quantile(
                triggered_table[f"area({skymap_area_cl})"], [0.9], method="inverted_cdf"
            ).item()
            if len(triggered_table) > 0
            else 0
        )

        perc_90_area_det = (
            np.quantile(
                triggered_table[f"area({skymap_area_cl})"],
                [0.9],
                weights=triggered_table["detection_probability_known_position"],
                method="inverted_cdf",
            ).item()
            if len(triggered_table) > 0
            else 0
        )

        # Horizontal orientation to align with main plot
        ax_right.hist(
            table[f"area({skymap_area_cl})"],
            bins=bins_area,
            orientation="horizontal",
            color="lightgray",
            alpha=0.6,
            edgecolor="gray",
            linewidth=1,
            zorder=1,
        )
        ax_right.hist(
            triggered_table[f"area({skymap_area_cl})"],
            bins=bins_area,
            orientation="horizontal",
            color="dodgerblue",
            alpha=0.7,
            edgecolor="darkblue",
            linewidth=1,
            zorder=2,
        )
        ax_right.hist(
            triggered_table[f"area({skymap_area_cl})"],
            bins=bins_area,
            weights=weights_detected,
            orientation="horizontal",
            color="red",
            alpha=0.8,
            edgecolor="darkred",
            linewidth=1.5,
            zorder=3,
        )

        # Draw HORIZONTAL lines for area percentiles
        ax_right.axhline(
            perc_90_area_all, linewidth=2, color="gray", alpha=0.8, zorder=4
        )
        ax_right.axhline(
            perc_90_area_trig, linewidth=2, color="dodgerblue", alpha=0.9, zorder=5
        )
        ax_right.axhline(
            perc_90_area_det, linewidth=2, color="red", alpha=0.9, zorder=6
        )

        ax_right.set_xlabel("Counts", fontsize=12, weight="bold")
        ax_right.set_yscale("log")
        ax_right.grid(True, alpha=0.3, linestyle="-", color="gray")
        ax_right.set_xlim(left=0)

        # ===================================================================
        # Add run label in top-right corner
        # ===================================================================
        fig.text(
            0.94,
            0.94,
            f"{mission.name.upper()}\n\n{run}",
            fontsize=18,
            weight="bold",
            ha="right",
            va="top",
            bbox={
                "facecolor": "white",
                "edgecolor": "black",
                "boxstyle": "round,pad=0.7",
                "linewidth": 2.5,
                "alpha": 1.0,
            },
        )

        # ===================================================================
        # Calculate and display summary statistics
        # ===================================================================
        total_events = len(table)
        n_triggered = len(triggered_table)
        n_detected_eff = weights_detected.sum()  # Effective number of detections
        perc_90_trig = (
            np.percentile(triggered_table["distance"], 90)
            if len(triggered_table) > 0
            else 0
        )

        # ===================================================================
        # Calculate and display summary statistics WITH percentiles table
        # ===================================================================
        total_events = len(table)
        n_triggered = len(triggered_table)
        n_detected_eff = weights_detected.sum()

        # Format statistics text WITH percentiles
        stats_text = (
            # f"Total: {total_events}\n"
            # f"Triggered: {n_triggered} ({100 * n_triggered / total_events:.1f}%)\n"
            # f"Detected (eff): {n_detected_eff:.0f}\n"
            # f"\n"
            f"90th percentiles:\n"
            f"All events: {perc_90_all:.0f}  Mpc\n"
            f"Triggered:  {perc_90_trig:.0f} Mpc\n"
            f"Detected:   {perc_90_det:.0f} Mpc"
        )

        # Statistics box with percentiles
        fig.text(
            0.115,
            0.94,
            stats_text,
            fontsize=9,  # Slightly smaller for more text
            ha="left",
            va="top",
            family="monospace",
            weight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "boxstyle": "round,pad=0.6",
                "alpha": 1.0,
                "linewidth": 1.2,
            },
        )

        # ===================================================================
        # Statistics box #2: Area percentiles (BOTTOM-RIGHT)
        # ===================================================================
        stats_text_area = (
            f"90th percentiles:\n"
            f"All events: {perc_90_area_all:.0f}  deg$^2$ \n"
            f"Triggered:  {perc_90_area_trig:.0f} deg$^2$ \n"
            f"Detected:   {perc_90_area_det:.0f}  deg$^2$"
        )

        fig.text(
            0.77,
            0.11,
            stats_text_area,
            fontsize=9,
            ha="left",
            va="bottom",
            family="monospace",
            weight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "boxstyle": "round,pad=0.6",
                "alpha": 1.0,
                "linewidth": 1.2,
            },
        )

        # ===================================================================
        # Save figure to PDF
        # ===================================================================
        # Join on the non-empty parts only: a mission without skygrid
        # variants (UVEX) would otherwise yield "uvex__area-distance".
        stem = "_".join(part for part in (mission.name, skygrid) if part)
        output_file = f"{outdir}/{stem}_area-distance_{run}.pdf"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        log.info(f" Saved: {output_file}")
        log.info(
            f" Total: {total_events} | Triggered: {n_triggered} | Detected (eff): {n_detected_eff:.1f}"
        )

        # Clean up to free memory
        if not show:
            plt.close(fig)
        else:
            plt.show()


# if __name__ == "__main__":
#     root = get_project_root()
#     events_file = root / "data" / "events.ecsv"
#     plot_area_distance(events_file)
