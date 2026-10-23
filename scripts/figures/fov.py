#!/usr/bin/env python
"""Figure of the ULTRASAT field of view over the HEALPix grid of the scheduler.

Usage
-----
::

    python scripts/figures/fov.py --outdir paper_outputs

Outputs ``fov_ultrasat_204.pdf`` and ``.png`` in ``--outdir``.
"""

import argparse
from pathlib import Path

import ligo.skymap.plot  # noqa: F401  (registers the "astro ..." projections)
import numpy as np
import regions
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy_healpix import HEALPix
from m4opt.fov import footprint_healpix
from m4opt.missions import ultrasat
from matplotlib import pyplot as plt

FOV_FACE = "#E8B84C"
FOV_EDGE = "#A87A1A"
GRID_COLOR = "#CCCCCC"
BG_COLOR = "#FDF9F0"

NSIDE = 128
RADIUS = 9 * u.deg


def customize_style():
    """Apply the figure style of the paper."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["figure.figsize"] = [5.0, 5.0]


def plot_boundaries(ax, hpx, ipix, **kwargs):
    """Draw the outline of each HEALPix pixel of ``ipix``."""
    lons, lats = hpx.boundaries_lonlat(ipix, 1)
    coords = np.moveaxis([lons.to_value(u.deg), lats.to_value(u.deg)], 0, -1)
    transform = ax.get_transform("world")
    for coord in coords:
        ax.add_patch(plt.Polygon(coord, transform=transform, **kwargs))


def plot_fov(center=None):
    """Draw the ULTRASAT field of view and the pixels it covers.

    Parameters
    ----------
    center : astropy.coordinates.SkyCoord, optional
        Boresight of the pointing; the origin by default.

    Returns
    -------
    matplotlib.figure.Figure
        The finished figure.
    """
    if center is None:
        center = SkyCoord(0 * u.deg, 0 * u.deg)
    hpx = HEALPix(nside=NSIDE, frame=ICRS())

    fig = plt.figure(tight_layout=True, facecolor="white")
    ax = fig.add_subplot(projection="astro zoom", center=center, radius=RADIUS)
    ax.set_facecolor(BG_COLOR)
    transform = ax.get_transform("world")

    # HEALPix grid, clipped to what the panel shows.
    visible = footprint_healpix(
        hpx,
        regions.PolygonSkyRegion(SkyCoord(*ax.wcs.calc_footprint().T, unit=u.deg)),
        center,
    )
    plot_boundaries(
        ax,
        hpx,
        visible,
        facecolor="none",
        edgecolor=GRID_COLOR,
        linewidth=0.4,
        zorder=1,
    )

    plot_boundaries(
        ax,
        hpx,
        footprint_healpix(hpx, ultrasat.fov, center),
        facecolor=FOV_FACE,
        edgecolor=FOV_FACE,
        alpha=0.45,
        linewidth=0.0,
        zorder=2,
    )
    ax.add_artist(
        ultrasat.fov.to_pixel(ax.wcs).as_artist(
            facecolor="none", edgecolor=FOV_EDGE, linewidth=2, zorder=3
        )
    )
    ax.plot(
        center.ra.deg,
        center.dec.deg,
        "+",
        color=FOV_EDGE,
        markersize=8,
        markeredgewidth=1.4,
        transform=transform,
        zorder=4,
    )
    ax.text(
        0.50,
        0.02,
        r"FOV $= 204\,\mathrm{deg}^2$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color=FOV_EDGE,
        bbox={
            "boxstyle": "round,pad=0.25",
            "fc": "white",
            "ec": FOV_EDGE,
            "alpha": 0.85,
            "linewidth": 0.8,
        },
    )

    for axis, label in (
        (ax.coords["ra"], "R.A. (deg)"),
        (ax.coords["dec"], "Dec. (deg)"),
    ):
        # Degrees rather than the hours the projection defaults to.
        axis.set_major_formatter("d.d")
        axis.set_ticks(spacing=5 * u.deg)
        axis.set_axislabel(label, fontsize=9, minpad=0.5)
        axis.set_ticklabel(size=8, color="#555555")
        axis.set_ticks_visible(True)
        axis.set_ticklabel_visible(True)

    ax.coords.frame.set_linewidth(0.8)
    ax.coords.frame.set_color("#AAAAAA")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--outdir", type=Path, default=Path("paper_outputs"), help="output directory"
    )
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    customize_style()
    fig = plot_fov()
    for suffix in ("pdf", "png"):
        fig.savefig(
            args.outdir / f"fov_ultrasat_204.{suffix}", bbox_inches="tight", dpi=300
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
