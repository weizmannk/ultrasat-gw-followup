#!/usr/bin/env python
r"""Figures of one M4OPT observing plan, as a still frame or an animation.

Two layouts share the same data products:

``zoom``    Zoomed sky map and timeline (style of the M4OPT talk figure).
            Colour map: exposure time required for a kilonova at the mean
            distance. Observed fields are shaded once per visit; the field
            being observed is outlined in black.

``allsky``  All-sky Mollweide map and timeline (style of ``m4opt animate``).
            Adds the field of regard: dark blue = never observable during
            the plan, light blue = not observable at the time shown, plus the
            Sun, Moon and Earth.

Timeline (both layouts): cumulative probability and area covered.
Light line = covered at least once; dark line = covered at least twice.

Examples
--------
    python plan_figures.py 721.ecsv --skymap 721.fits --layout zoom \
        --center "0d -14d" --radius 47deg --aspect 0.66 \
        --time 4.75h --still plan_zoom.pdf --movie plan_zoom.mp4

    python plan_figures.py 721.ecsv --skymap 721.fits --layout allsky \
        --time 4.75h --still plan_allsky.pdf --movie plan_allsky.mp4

Exposure times, footprints, coverage and field of regard are computed with the
same M4OPT functions as ``m4opt animate``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

import ligo.skymap.plot  # noqa: F401  (registers the "astro ..." projections)
import numpy as np
import synphot
from astropy import units as u
from astropy.coordinates import ICRS, Distance, SkyCoord, get_body
from astropy.table import QTable
from astropy.time import Time
from astropy_healpix import HEALPix
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from ligo.skymap.plot.marker import earth, moon, sun
from ligo.skymap.plot.poly import cut_prime_meridian
from ligo.skymap.postprocess import find_greedy_credible_levels
from m4opt import missions
from m4opt.fov import footprint, footprint_healpix
from m4opt.synphot import observing
from m4opt.synphot.extinction import DustExtinction
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from regions import Regions


# --------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Style:
    prob: str  # probability axis and 90% contour
    area: str  # area axis and footprints
    cmap: str  # exposure-time map
    cmap_alpha: float
    outside_avg: str = "#1f78b4"  # never observable during the plan
    outside_now: str = "#a6cee3"  # not observable at the time shown


STYLES = {
    "zoom": Style(prob="#1f77ff", area="#ff4d00", cmap="viridis", cmap_alpha=1.0),
    "allsky": Style(prob="#33a02c", area="#e31a1c", cmap="binary", cmap_alpha=0.5),
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
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

    @classmethod
    def read(cls, path: str, skymap: str | None = None) -> Plan:
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
        )

    def hours(self, time: Time) -> np.ndarray:
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


def wrap_ra(ra_deg: np.ndarray, center_deg: float) -> np.ndarray:
    """Keep RA continuous around the zoom centre (fields crossing 0h)."""
    return (ra_deg - center_deg + 180) % 360 - 180 + center_deg


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
class PlanFigure:
    def __init__(
        self,
        plan,
        products,
        layout,
        center=None,
        radius=None,
        aspect=1.0,
        vmax=None,
        dpi=300,
    ):
        self.plan, self.prod, self.layout = plan, products, layout
        self.style = STYLES[layout]
        plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})

        if layout == "zoom":
            self.fig = plt.figure(figsize=(4.2, 4.7), dpi=dpi)
            map_rect, cbar_rect, time_rect = (
                (0.12, 0.415, 0.84, 0.52),
                (0.12, 0.315, 0.84, 0.02),
                (0.12, 0.075, 0.72, 0.14),
            )
            if center is None or radius is None:
                center, radius = auto_zoom(plan)
            self.center = center
            self.ax = self.fig.add_axes(
                map_rect, projection="astro degrees zoom", center=center, radius=radius
            )
            self._format_zoom_axes(aspect)
        else:
            self.fig = plt.figure(figsize=(6.4, 4.8), dpi=dpi)
            map_rect, cbar_rect, time_rect = (
                (0.05, 0.40, 0.90, 0.48),
                (0.15, 0.32, 0.70, 0.025),
                (0.12, 0.08, 0.74, 0.15),
            )
            # Centre the Mollweide map on the observed fields (RA only).
            self.center = (
                SkyCoord(auto_zoom(plan)[0].ra, 0 * u.deg) if center is None else center
            )
            self.ax = self.fig.add_axes(
                map_rect, projection="astro hours mollweide", center=self.center
            )
            self.ax.grid()
            self._add_allsky_legends()

        self.world = self.ax.get_transform("world")
        self._draw_exposure_map(cbar_rect, vmax)
        self.ax.contour_hpx(
            products.credible_level,
            levels=[0.9],
            nested=True,
            colors=[self.style.prob],
            linewidths=1.2,
            zorder=4,
        )
        self.fills, self.outlines = self._make_footprints()
        self._draw_timeline(time_rect)

        self.for_now = None
        if layout == "allsky":
            self.times = self._time_grid(5 * u.min)
            self.observable = field_of_regard(plan, products.hpx, self.times)
            self.ax.contourf_hpx(
                self.observable.any(axis=0),
                nested=True,
                levels=[-1, 0.5],
                colors=[self.style.outside_avg],
                zorder=1.1,
            )
            self.bodies = {
                name: self.ax.plot(0, 0, transform=self.world, zorder=10000, **kw)[0]
                for name, kw in {
                    "earth": {"marker": earth, "mec": "black", "mfc": "none"},
                    "sun": {"marker": sun, "mec": "black", "mfc": "none"},
                    "moon": {"marker": moon(-115), "mfc": "black", "mec": "none"},
                }.items()
            }

    # ---- construction helpers ------------------------------------------------
    def _format_zoom_axes(self, aspect):
        ax = self.ax
        ax.grid(color="0.85", lw=0.5)
        for key, pos in (("ra", "t"), ("dec", "l")):
            ax.coords[key].set_major_formatter("d")
            ax.coords[key].set_ticks_position(pos)
            ax.coords[key].set_ticklabel_position(pos)
            ax.coords[key].set_axislabel("")
        if aspect < 1:  # crop in Dec: wide panel for fields spread in RA
            y0, y1 = ax.get_ylim()
            ym, half = 0.5 * (y0 + y1), 0.5 * (y1 - y0) * aspect
            ax.set_ylim(ym - half, ym + half)
            ax.set_anchor("S")

    def _add_allsky_legends(self):
        s = self.style
        self.fig.legend(
            [
                Patch(facecolor=s.area, alpha=1 / self.plan.visits),
                Patch(facecolor="none", edgecolor=s.prob),
            ],
            ["observation footprints", "90% credible region"],
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
        )
        self.fig.legend(
            [Patch(facecolor=s.outside_avg), Patch(facecolor=s.outside_now)],
            ["during the whole plan", "at the time shown"],
            title="not observable",
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
        )

    def _draw_exposure_map(self, cbar_rect, vmax):
        durations = self.plan.obs["duration"].to_value(u.min)
        vmax = vmax or 10 * np.ceil(durations.max() / 10)
        im = self.ax.imshow_hpx(
            self.prod.exptime_min,
            nested=True,
            cmap=self.style.cmap,
            vmin=0,
            vmax=vmax,
            alpha=self.style.cmap_alpha,
        )
        cbar = self.fig.colorbar(
            im, cax=self.fig.add_axes(cbar_rect), orientation="horizontal"
        )
        cbar.set_label("Required exposure time (min)")
        cbar.ax.xaxis.set_label_position("top")

    def _polygons(self, region):
        subregions = region.regions if isinstance(region, Regions) else [region]
        for r in subregions:
            ra, dec = r.vertices.ra, r.vertices.dec
            if self.layout == "zoom":
                yield np.column_stack((wrap_ra(ra.deg, self.center.ra.deg), dec.deg))
            else:  # split polygons that cross the map edge (centre RA + 180 deg)
                edge = self.center.ra.rad + np.pi
                for v in cut_prime_meridian(
                    np.column_stack(((ra.rad - edge) % (2 * np.pi), dec.rad))
                ):
                    yield np.column_stack(
                        (np.rad2deg(v[:, 0] + edge), np.rad2deg(v[:, 1]))
                    )

    def _make_footprints(self):
        fills, outlines = [], []
        regions = footprint(
            self.plan.mission.fov, self.plan.obs["target_coord"], self.plan.obs["roll"]
        )
        for region in regions:
            polys = list(self._polygons(region))
            fills.append(
                [
                    self.ax.add_patch(
                        plt.Polygon(
                            v,
                            transform=self.world,
                            facecolor=self.style.area,
                            alpha=1 / self.plan.visits,
                            lw=0,
                            zorder=2,
                            visible=False,
                        )
                    )
                    for v in polys
                ]
            )
            outlines.append(
                [
                    self.ax.add_patch(
                        plt.Polygon(
                            v,
                            transform=self.world,
                            facecolor="none",
                            edgecolor="black",
                            lw=1.0,
                            zorder=3,
                            visible=False,
                        )
                    )
                    for v in polys
                ]
            )
        return fills, outlines

    def _draw_timeline(self, rect):
        plan, prod, s = self.plan, self.prod, self.style
        axp = self.fig.add_axes(rect)
        axa = axp.twinx()
        start, end = (
            plan.hours(plan.obs["start_time"]),
            plan.hours(plan.obs["end_time"]),
        )
        for i in range(plan.visits):
            a = (i + 1) / plan.visits  # light = >= 1 visit, dark = all visits
            cp, ca = (np.asarray(to_rgb(c)) * a + (1 - a) for c in (s.prob, s.area))
            axp.hlines(prod.covered_prob[:, i], start, end, cp, lw=1.5)
            axa.hlines(prod.covered_area[:, i], start, end, ca, lw=1.5)
        axp.axvspan(0, plan.delay.to_value(u.hour), color="0.88", lw=0)
        axp.set(
            xlim=(0, plan.deadline.to_value(u.hour)),
            ylim=(0, 1),
            xlabel="time in hours since event",
        )
        axa.set_ylim(0, None)
        axp.set_ylabel("probability", color=s.prob)
        axa.set_ylabel("area (deg$^2$)", color=s.area)
        axp.tick_params(axis="y", colors=s.prob)
        axa.tick_params(axis="y", colors=s.area)
        self.now_line = axp.axvline(0, color="black", lw=1)
        self.now_text = axp.text(
            0,
            1.02,
            "now",
            ha="center",
            va="bottom",
            transform=blended_transform_factory(axp.transData, axp.transAxes),
        )

    def _time_grid(self, step):
        plan = self.plan
        n = int(np.floor((plan.deadline / step).decompose().value)) + 1
        return plan.event_time + np.arange(n) * step

    # ---- frame update ----------------------------------------------------------
    def draw(self, time: Time):
        h = self.plan.hours(time)
        self.now_line.set_xdata([h])
        self.now_text.set_x(h)
        started = self.plan.obs["start_time"] <= time
        running = started & (self.plan.obs["end_time"] > time)
        for fills, outlines, s, r in zip(self.fills, self.outlines, started, running):
            for p in fills:
                p.set_visible(bool(s))
            for p in outlines:
                p.set_visible(bool(r))
        if self.layout == "allsky":
            self._draw_allsky_state(time)

    def _draw_allsky_state(self, time):
        k = int(np.argmin(np.abs((self.times - time).to_value(u.s))))
        if self.for_now is not None:
            self.for_now.remove()
        self.for_now = self.ax.contourf_hpx(
            self.observable[k],
            nested=True,
            levels=[-1, 0.5],
            colors=[self.style.outside_now],
            zorder=1.05,
        )
        location = self.plan.mission.observer_location(self.times[k])
        for name, artist in self.bodies.items():
            c = get_body(name, self.times[k], location)
            artist.set_data([c.ra.deg], [c.dec.deg])

    # ---- output ------------------------------------------------------------------
    def save_still(self, path, time):
        self.draw(time)
        self.fig.savefig(path)

    def save_movie(self, path, step, fps=12):
        writer = (
            FFMpegWriter(fps=fps, bitrate=4000)
            if Path(path).suffix == ".mp4"
            else PillowWriter(fps=fps)
        )
        FuncAnimation(self.fig, self.draw, frames=list(self._time_grid(step))).save(
            path, writer=writer, dpi=min(self.fig.dpi, 200)
        )


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples" + __doc__.split("Examples")[1],
    )
    p.add_argument("schedule", help="schedule (.ecsv) written by `m4opt schedule`")
    p.add_argument(
        "--skymap", help="sky map (.fits); default: path stored in the schedule"
    )
    p.add_argument("--layout", choices=STYLES, default="zoom")
    p.add_argument(
        "--center",
        type=SkyCoord,
        help='map centre, e.g. "0d -14d" (default: centre of the observed fields)',
    )
    p.add_argument(
        "--radius", type=u.Quantity, help="zoom radius, e.g. 47deg (default: automatic)"
    )
    p.add_argument(
        "--aspect",
        type=float,
        default=1.0,
        help="zoom panel height/width (<1 crops in Dec)",
    )
    p.add_argument(
        "--time",
        type=u.Quantity,
        default=None,
        help="time of the still frame since the event (default: deadline)",
    )
    p.add_argument("--still", help="output still frame (.pdf, .png)")
    p.add_argument("--movie", help="output animation (.mp4, .gif)")
    p.add_argument(
        "--step", type=u.Quantity, default=2.5 * u.min, help="animation time step"
    )
    p.add_argument("--vmax", type=float, help="colour-bar maximum (min)")
    p.add_argument("--dpi", type=float, default=300)
    args = p.parse_args(argv)
    if not (args.still or args.movie):
        p.error("give --still and/or --movie")
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = Plan.read(args.schedule, args.skymap)
    figure = PlanFigure(
        plan,
        Products.compute(plan),
        args.layout,
        args.center,
        args.radius,
        args.aspect,
        args.vmax,
        args.dpi,
    )
    if args.movie:
        figure.save_movie(args.movie, args.step)
    if args.still:
        figure.save_still(
            args.still,
            plan.event_time + (args.time if args.time is not None else plan.deadline),
        )


if __name__ == "__main__":
    main()
