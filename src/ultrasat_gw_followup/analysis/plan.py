"""Quantities of one M4OPT observing plan, read back from its schedule.

A schedule written by ``m4opt schedule`` records the observations and, in its
metadata, every setting used to compute them. :class:`Plan` reads both, and
:class:`Products` derives from them the four maps the plan is built on: the
localization probability, its greedy credible levels, the exposure time a
kilonova would require in each direction, and the coverage the plan achieves.

They are computed with the same M4OPT functions as ``m4opt animate``, so a
figure or a tutorial drawn from them shows what the scheduler optimized, not
an approximation of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate

import numpy as np
import synphot
from astropy import units as u
from astropy.coordinates import ICRS, Distance, SkyCoord
from astropy.table import QTable
from astropy.time import Time
from astropy_healpix import HEALPix
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import find_greedy_credible_levels
from m4opt import missions
from m4opt.fov import footprint_healpix
from m4opt.synphot import observing
from m4opt.synphot.extinction import DustExtinction


@dataclass
class Plan:
    """Observations of one schedule and the settings used to compute it."""

    obs: QTable
    mission: missions.Mission
    event_time: Time
    delay: u.Quantity
    deadline: u.Quantity
    visits: int
    nside: int
    bandpass: str
    snr: float
    absmag_mean: float
    skymap: str
    skygrid: str | None  # named grid of candidate pointings, if the mission has several

    @property
    def candidates(self) -> SkyCoord:
        """The pointings the scheduler had to choose from."""
        grid = self.mission.skygrid
        return grid[self.skygrid] if isinstance(grid, dict) else grid

    @classmethod
    def read(cls, path: str, skymap: str | None = None) -> Plan:
        """Read a schedule and the arguments its metadata records.

        Parameters
        ----------
        path : str
            Schedule (``.ecsv``) written by ``m4opt schedule``.
        skymap : str, optional
            Sky map to use instead of the path stored in the schedule, which
            is the one of the machine the plan was computed on.

        Returns
        -------
        Plan
        """
        table = QTable.read(path, format="ascii.ecsv")
        args = table.meta["args"]
        obs = table[table["action"] == "observe"]
        obs["end_time"] = obs["start_time"] + obs["duration"]
        bandpass = args["bandpass"]
        return cls(
            obs=obs,
            mission=getattr(missions, args["mission"]),
            event_time=Time(args["event_time"]),
            delay=u.Quantity(args["delay"]),
            deadline=u.Quantity(args["deadline"]),
            visits=int(args["visits"]),
            nside=int(args["nside"]),
            bandpass=bandpass if isinstance(bandpass, str) else bandpass[0],
            snr=float(args["snr"]),
            absmag_mean=float(args["absmag_mean"]),
            skymap=skymap or args["skymap"],
            skygrid=args.get("skygrid"),
        )

    def hours(self, time: Time) -> np.ndarray:
        """Hours elapsed since the merger."""
        return (time - self.event_time).to_value(u.hour)


@dataclass
class Products:
    """Quantities derived once from the plan and the sky map."""

    hpx: HEALPix
    prob: np.ndarray  # probability per pixel
    credible_level: np.ndarray  # greedy credible level per pixel
    exptime_min: np.ndarray  # required exposure time per pixel (min)
    covered_prob: np.ndarray  # (n_obs, visits) cumulative probability
    covered_area: np.ndarray  # (n_obs, visits) cumulative area (deg2)

    @classmethod
    def compute(cls, plan: Plan) -> Products:
        hpx = HEALPix(plan.nside, frame=ICRS(), order="nested")
        moc = read_sky_map(plan.skymap, moc=True)
        prob = rasterize(moc["UNIQ", "PROBDENSITY"], hpx.level)["PROB"]

        # Exposure time for a source of absolute magnitude absmag_mean at the
        # mean distance, evaluated at the start of observations.
        t0 = plan.event_time + plan.delay
        source = synphot.SourceSpectrum(
            synphot.ConstFlux1D(
                plan.absmag_mean * u.ABmag
                + Distance(moc.meta["distmean"] * u.Mpc).distmod
            )
        )
        with observing(
            observer_location=plan.mission.observer_location(t0),
            target_coord=hpx.healpix_to_skycoord(np.arange(hpx.npix)),
            obstime=t0,
        ):
            exptime = plan.mission.detector.get_exptime(
                plan.snr, source * DustExtinction(), plan.bandpass
            ).to_value(u.min)

        # Cumulative coverage after each observation, for >= 1, ..., >= visits.
        n_visit = np.arange(plan.visits)[:, np.newaxis]
        covered_prob = np.empty((len(plan.obs), plan.visits))
        covered_area = np.empty_like(covered_prob)
        pixels = footprint_healpix(
            hpx, plan.mission.fov, plan.obs["target_coord"], plan.obs["roll"]
        )
        for k, pix in enumerate(accumulate(pixels, lambda *a: np.concatenate(a))):
            visited = np.bincount(pix, minlength=hpx.npix) > n_visit
            covered_prob[k] = (prob * visited).sum(axis=1)
            covered_area[k] = visited.sum(axis=1) * hpx.pixel_area.to_value(u.deg**2)

        return cls(
            hpx=hpx,
            prob=prob,
            credible_level=find_greedy_credible_levels(prob),
            exptime_min=exptime,
            covered_prob=covered_prob,
            covered_area=covered_area,
        )


def field_of_regard(plan: Plan, hpx: HEALPix, times: Time) -> np.ndarray:
    """Boolean array (n_times, n_pix): True where the target is observable."""
    return plan.mission.constraints(
        plan.mission.observer_location(times)[:, np.newaxis],
        hpx.healpix_to_skycoord(np.arange(hpx.npix)),
        times[:, np.newaxis],
    )


def auto_zoom(plan: Plan) -> tuple[SkyCoord, u.Quantity]:
    """Centre on the observed fields; radius covers them with a margin."""
    coords = plan.obs["target_coord"]
    xyz = np.asarray(
        getattr(coords.cartesian.xyz, "unmasked", coords.cartesian.xyz).value
    )
    x, y, z = xyz.mean(axis=1)
    center = SkyCoord(
        np.degrees(np.arctan2(y, x)) % 360,
        np.degrees(np.arctan2(z, np.hypot(x, y))),
        unit="deg",
        frame="icrs",
    )
    sep = float(
        np.max(
            getattr(
                coords.separation(center), "unmasked", coords.separation(center)
            ).deg
        )
    )
    return center, (sep + 0.75 * plan.mission.fov.width.to_value(u.deg)) * u.deg
