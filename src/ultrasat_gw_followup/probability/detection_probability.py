# This probability is from https://github.com/m4opt/m4opt-paper

import warnings

import numpy as np
import synphot
from astropy import units as u
from astropy.coordinates import ICRS, Distance
from astropy.table import unique
from astropy_healpix import HEALPix, nside_to_level
from ligo.skymap import distance
from ligo.skymap.bayestar import rasterize
from m4opt import missions
from m4opt.fov import footprint_healpix
from m4opt.synphot import observing
from m4opt.synphot.extinction import DustExtinction
from scipy import stats

warnings.filterwarnings("ignore", ".*Wswiglal-redir-stdio.*")
warnings.filterwarnings("ignore", ".*dubious year.*")
warnings.filterwarnings("ignore", ".*polar motions.*")


# -- helpers ------------------------------------------------------------
def _limmag_per_row(fields, mission, plan_args):
    """Limiting magnitude of every row of ``fields``, in one vectorized call.

    The mission background model is evaluated from the ``observer_location`` and
    ``obstime`` of the ``observing()`` context. For ULTRASAT this already
    includes the position-dependent Cerenkov / AE8 term (m4opt >= 2.13), so no
    per-field loop is needed — this was verified to match the old field-by-field
    result bit-for-bit.

    Parameters
    ----------
    fields : astropy.table.QTable
        Observation rows, carrying ``observer_location``, ``target_coord``,
        ``start_time`` and ``duration``.
    mission : m4opt.missions.Mission
        Mission whose detector model sets the sensitivity.
    plan_args : dict
        Plan metadata; ``snr`` and ``bandpass`` are used here.

    Returns
    -------
    astropy.units.Quantity
        Limiting apparent magnitude, one per row.
    """
    with observing(
        observer_location=fields["observer_location"],
        target_coord=fields["target_coord"],
        obstime=(fields["start_time"] + 0.5 * fields["duration"]),
    ):
        spectrum = synphot.SourceSpectrum(
            synphot.ConstFlux1D, amplitude=0 * u.ABmag
        ) * synphot.SpectralElement(DustExtinction())
        return mission.detector.get_limmag(
            plan_args["snr"], fields["duration"], spectrum, plan_args["bandpass"]
        )


def _compute_limmag(fields, mission, plan_args):
    """Best limiting magnitude over all observed fields."""
    return _limmag_per_row(fields, mission, plan_args).max()


def _fields_covering_target(plan, event_row, plan_args, deduplicate):
    """Observation rows whose footprint contains the true source position.

    Parameters
    ----------
    plan : astropy.table.QTable
        Schedule as written by m4opt.
    event_row : astropy.table.Row
        Simulated event, read for ``longitude`` and ``latitude``.
    plan_args : dict
        Plan metadata; ``nside`` and ``mission`` are used here.
    deduplicate : bool
        If True, keep one row per distinct pointing. ``astropy.table.unique``
        keeps the *first* row of each group, which is the first visit and not
        necessarily the longest one, so the metric built on top of this is a
        maximum over first visits. If False, every visit is kept, which is what
        a criterion requiring detection in more than one visit needs.

    Returns
    -------
    fields : astropy.table.QTable or None
        The covering rows, or None when no field covers the position.
    mission : m4opt.missions.Mission
        Mission named by ``plan_args``.
    """
    hpx = HEALPix(nside=plan_args["nside"], order="nested", frame=ICRS())
    mission = getattr(missions, plan_args["mission"])

    observations = observed_fields(plan)
    coords = observations["target_coord"].to_table()
    coords["i"] = np.arange(len(coords))
    if deduplicate:
        i = np.sort(unique(coords, keys=["ra", "dec"])["i"])
    else:
        i = np.asarray(coords["i"])
    fields = observations[i]

    target_ipix = hpx.lonlat_to_healpix(
        event_row["longitude"] * u.rad, event_row["latitude"] * u.rad
    )
    target_in_field = [
        target_ipix in footprint
        for footprint in footprint_healpix(
            hpx, mission.fov, fields["target_coord"], fields["roll"]
        )
    ]
    fields = fields[target_in_field]
    return (fields if len(fields) else None), mission


def _visit_groups(fields):
    """Key identifying repeated visits to the same pointing, one per row.

    ``field_id`` names the sky grid tile the observation belongs to. m4opt
    writes it from 2.13 on, and :mod:`ultrasat_gw_followup.workflow.postprocess`
    already counts fields with it, so it is the authoritative key here too.
    Older schedules carry no such column, and those fall back to the pointing
    coordinates. The two give the same partition on the schedules we have, but
    the identifier does not rely on two visits producing bit-identical floats.

    Parameters
    ----------
    fields : astropy.table.QTable
        Observation rows.

    Returns
    -------
    numpy.ndarray
        Group key per row; rows sharing a key are visits of one field.
    """
    if "field_id" in fields.colnames:
        return np.asarray(fields["field_id"])

    coords = fields["target_coord"]
    _, group = np.unique(
        np.stack([coords.ra.deg, coords.dec.deg], axis=-1),
        axis=0,
        return_inverse=True,
    )
    return np.ravel(group)


def _probability_brighter_than(limmag, event_row, plan_args):
    """Probability that the kilonova is brighter than a limiting magnitude.

    The absolute magnitude is Gaussian, so this is just its CDF evaluated at
    the limit converted to an absolute magnitude at the source distance.
    """
    lim_absmag = limmag - Distance(event_row["distance"] * u.Mpc).distmod
    return stats.norm(
        loc=plan_args["absmag_mean"], scale=plan_args["absmag_stdev"]
    ).cdf(lim_absmag.to_value(u.mag))


def _any_masked(column):
    """Whether ``column`` actually has masked entries.

    SkyCoord and EarthLocation carry a structured mask with one field per
    axis, and reductions such as ``any()`` do not work on those directly.
    """
    mask = getattr(column, "mask", None)
    if mask is None:
        return False
    if getattr(mask, "dtype", None) is not None and mask.dtype.names:
        return any(mask[name].any() for name in mask.dtype.names)
    return bool(mask.any())


def observed_fields(plan):
    """Select the observing rows of a schedule and resolve their masks.

    ``QTable.filled()`` fills every column at once, which raises on a masked
    Time column because astropy has no fill value it can convert to a time.
    m4opt schedules do carry a masked ``start_time`` -- the mask marks the
    rows that are not observations -- so only call ``filled()`` when this
    selection genuinely still holds masked entries. When it does not, the
    values are already the ones ``filled()`` would have returned.
    """
    observations = plan[plan["action"] == "observe"]
    if any(_any_masked(column) for column in observations.columns.values()):
        return observations.filled()
    return observations


# -- main ------------------------------------------------------------
def get_detection_probability_known_position(plan, event_row, plan_args):
    """Probability that the kilonova is above threshold in the deepest field.

    One row per distinct pointing is kept, and the limiting magnitude is the
    best of those. This answers "was the source bright enough to be above
    threshold somewhere in the plan", not "would it have been identified as a
    transient" -- see :func:`get_detection_probability_two_visits` for the
    stricter criterion.

    Parameters
    ----------
    plan : astropy.table.QTable
        Schedule as written by m4opt.
    event_row : astropy.table.Row
        Simulated event, read for its true position and distance.
    plan_args : dict
        Plan metadata.

    Returns
    -------
    float
        Detection probability; 0 when the plan is empty or no field covers the
        true position.
    """
    if len(plan) == 0:
        return 0

    fields, mission = _fields_covering_target(
        plan, event_row, plan_args, deduplicate=True
    )
    if fields is None:
        return 0

    limmag = _compute_limmag(fields, mission, plan_args)
    return _probability_brighter_than(limmag, event_row, plan_args)


def get_detection_probability_two_visits(plan, event_row, plan_args):
    """Probability that the kilonova is above threshold in *every* visit.

    Rejecting moving Solar System objects, and so identifying the source as a
    transient at all, needs it detected in more than one visit. The absolute
    magnitude is a **single draw** X ~ N(absmag_mean, absmag_stdev): both
    visits look at the same source, so their detections are not independent
    events to be multiplied. Visit k detects iff X is brighter than that
    visit's limiting absolute magnitude, so every visit of a field detects iff
    X is brighter than the faintest limit among them. Hence the min over the
    visits of a field, then the max over the fields that cover the position:

    .. math::

        P = \\Phi\\!\\left(
            \frac{\\max_\\mathrm{fields} \\min_\\mathrm{visits} m
                   - \\mu_X}{\\sigma_X}\right).

    No integration and no independence assumption is involved. Multiplying the
    per-visit probabilities would be wrong: it would treat one source as two
    draws.

    This is a lower bound on what a real search achieves, because it ignores
    coaddition: a source below threshold in each single visit can still be
    recovered in their sum. It is an upper bound on nothing.

    Parameters
    ----------
    plan : astropy.table.QTable
        Schedule as written by m4opt.
    event_row : astropy.table.Row
        Simulated event, read for its true position and distance.
    plan_args : dict
        Plan metadata.

    Returns
    -------
    float
        Detection probability; 0 when the plan is empty or no field covers the
        true position.
    """
    if len(plan) == 0:
        return 0

    fields, mission = _fields_covering_target(
        plan, event_row, plan_args, deduplicate=False
    )
    if fields is None:
        return 0

    limmags = _limmag_per_row(fields, mission, plan_args)

    # Faintest limit within each field, then the best field.
    group = _visit_groups(fields)
    limmag = max(limmags[group == g].min() for g in np.unique(group))
    return _probability_brighter_than(limmag, event_row, plan_args)


def get_detection_probability_unknown_position(plan, skymap_moc, plan_args):
    """
    Estimate the detection probability of a transient with an uncertain sky position,
    described by a gravitational wave probability skymap.

    This function computes the overall detection probability by integrating over all
    possible sky positions and distances from the skymap. For each sky location, it
    considers the total exposure from the observation plan, computes the limiting
    magnitude, and estimates the chance that a transient with an assumed absolute
    magnitude distribution would be detectable at that distance. The final probability
    is the sum of the detection probabilities weighted by the skymap's localization
    probabilities.

    Parameters
    ----------
    plan : astropy.table.Table
        Table of scheduled observations, including pointing coordinates, times, and durations.
    skymap_moc :
        Gravitational  wave localization skymap, encoding the sky probability
        and distance estimates per pixel.
    plan_args : dict
        Dictionary of observation and mission parameters (e.g., nside, mission name, SNR threshold,
        bandpass, mean and standard deviation of absolute magnitude).

    Returns
    -------
    probability : float
        Integrated probability that the transient would be detected by the observation plan,
        marginalized over the skymap localization.
    """

    if len(plan) == 0:
        return 0

    hpx = HEALPix(nside=plan_args["nside"], order="nested", frame=ICRS())
    mission = getattr(missions, plan_args["mission"])

    skymap = rasterize(skymap_moc, order=nside_to_level(plan_args["nside"]))

    observations = observed_fields(plan)
    coords = observations["target_coord"].to_table()
    coords["i"] = np.arange(len(coords))
    i = np.sort(unique(coords, keys=["ra", "dec"])["i"])
    fields = observations[i]

    durations = np.zeros(hpx.npix)
    for ipix, duration in zip(
        footprint_healpix(hpx, mission.fov, fields["target_coord"], fields["roll"]),
        fields["duration"].to_value(u.s),
    ):
        durations[ipix] = np.maximum(durations[ipix], duration)

    skymap["duration"] = durations * u.s
    skymap["ipix"] = np.arange(hpx.npix)

    skymap = skymap[durations > 0]

    with observing(
        observer_location=plan["observer_location"][0],
        target_coord=hpx.healpix_to_skycoord(skymap["ipix"]),
        obstime=plan["start_time"][0],
    ):
        # Cerenkov / AE8 background is part of the mission model and is evaluated
        # from this observing() context (m4opt >= 2.13).
        spectrum = synphot.SourceSpectrum(
            synphot.ConstFlux1D, amplitude=0 * u.ABmag
        ) * synphot.SpectralElement(DustExtinction())
        skymap["limmag"] = mission.detector.get_limmag(
            plan_args["snr"], skymap["duration"], spectrum, plan_args["bandpass"]
        )

    skymap["limmag"][np.isnan(skymap["limmag"])] = -np.inf * u.mag

    distmean, diststd, _ = distance.parameters_to_moments(
        skymap["DISTMU"], skymap["DISTSIGMA"]
    )
    sigma2_log = np.log1p(np.square(diststd / distmean))
    logdistsigma = np.sqrt(sigma2_log)
    logdistmu = np.log(distmean) - 0.5 * sigma2_log

    absmagmu = plan_args["absmag_mean"]
    absmagsigma = plan_args["absmag_stdev"]
    a = 5 / np.log(10)
    appmagmu = absmagmu + a * logdistmu + 25
    appmagsigma = np.sqrt(np.square(absmagsigma) + np.square(a * logdistsigma))
    skymap["appmagmu"] = appmagmu
    skymap["appmagsigma"] = appmagsigma
    return (
        skymap["PROB"]
        * stats.norm(loc=skymap["appmagmu"], scale=skymap["appmagsigma"]).cdf(
            skymap["limmag"]
        )
    ).sum()
