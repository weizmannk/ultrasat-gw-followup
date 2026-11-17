#!/usr/bin/env python
"""Figure of the generic grid of ULTRASAT pointings.

ULTRASAT can slew to any observable position after a trigger, so the grid of
candidate pointings is built from the field of view alone. The circle inscribed
in the square field stays inside the field at any roll angle, so a grid whose
centres lie within that radius of every position covers the whole sky at any
roll. The grid is the smallest geodesic grid of M4OPT that meets the condition.

The largest distance from a position to the nearest centre is the number the
caption quotes. It is computed exactly, from the spherical Voronoi diagram of
the centres.

The grid is derived here from the field of view, then checked against the
mission's own ``geodesic`` grid when the installed M4OPT defines one. The two
must agree: the scheduling runs read their grid from the
mission, and the figure has to show what they used.

Usage
-----
::
    # By default, the derived geodesic grid is drawn.
    python scripts/figures/sky_grid.py --outdir paper_outputs/

    # To draw the mission's own grids instead, repeat --skygrid for each one.
    python scripts/figures/sky_grid.py --outdir paper_outputs/ --skygrid allsky --skygrid non-overlap


Everything goes to a ``skygrid`` directory under ``--outdir``, one file per
panel, so that the document decides the layout:

``<grid>_fields.pdf``
    The field of view at every pointing, on a globe, at one roll angle.
    ``--gif`` animates it over roll angles from 0 to 90 degrees, which covers
    every orientation of a square.
``<grid>_coverage.pdf``
    Over the whole sky, how many inscribed circles contain each position. Zero
    means the grid leaves that position uncovered at some roll angle.

``--skygrid`` draws the mission's own grids instead, for comparison: ``allsky``,
the All-Sky Survey grid of 361 fields, and ``non-overlap``, the Low-Cadence
Survey grid of 240 that tiles the sky without redundancy.
"""

import argparse
from importlib.metadata import version
from pathlib import Path

import ligo.skymap.plot  # noqa: F401  (registers the "astro ..." projections)
import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy_healpix import HEALPix
from m4opt import skygrid
from m4opt.fov import footprint
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, ListedColormap
from regions import PolygonSkyRegion
from scipy.spatial import SphericalVoronoi

FULL_SKY = (4 * np.pi * u.sr).to(u.deg**2)

FIELD_EDGE = "#A87A1A"
#: Ink, not the darkest step: centres stay visible where four circles overlap.
CENTER_COLOR = "#2B2B28"
#: Coverage, from none to four circles. A gap is cool rather than red, which
#: is indistinguishable from deep ochre under protanopia.
COVERAGE_COLORS = ["#3B4CC0", "#F2DFA8", "#E8BC5E", "#B9771B", "#6B4410"]

ROLL = 45 * u.deg
GIF_ROLLS = np.arange(0, 90, 5) * u.deg
NSIDE = 128


def customize_style():
    """Apply the figure style of the paper."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def unit_vectors(coords):
    """Unit vectors of ``coords``, as an array of shape (n, 3)."""
    xyz = coords.cartesian.xyz.value.T
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


def covering_radius(centers):
    """Largest distance from any position on the sky to the nearest centre.

    The maximum falls on a vertex of the spherical Voronoi diagram of the
    centres, which makes it exact.

    Parameters
    ----------
    centers : astropy.coordinates.SkyCoord
        Grid centres.

    Returns
    -------
    astropy.units.Quantity
    """
    xyz = unit_vectors(centers)
    vertices = SphericalVoronoi(xyz).vertices
    nearest = np.arccos(np.clip(vertices @ xyz.T, -1, 1)).min(axis=1)
    return (nearest.max() * u.rad).to(u.deg)


def inscribed_radius(fov):
    """Radius of the circle inscribed in a rectangular field of view."""
    return 0.5 * min(fov.width, fov.height).to(u.deg)


def hexagon_area(radius):
    """Area of the regular hexagon inscribed in a circle of ``radius``.

    Hexagons are the largest cells that tile the plane while staying inside a
    circle of this radius, so no covering grid has fewer cells than
    ``FULL_SKY / hexagon_area(radius)``. That bound is where the search below
    starts.
    """
    return (1.5 * np.sqrt(3) * radius**2).to(u.deg**2)


def smallest_covering_grid(radius):
    """Smallest geodesic grid whose covering radius is at most ``radius``.

    Parameters
    ----------
    radius : astropy.units.Quantity
        Radius of the circle inscribed in the field of view.

    Returns
    -------
    astropy.coordinates.SkyCoord
        Grid centres.
    """
    area = hexagon_area(radius)
    while True:
        grid = skygrid.geodesic(area, class_="III", base="icosahedron")
        if covering_radius(grid) <= radius:
            return grid
        # Ask for one more vertex: the next geodesic grid up.
        area = FULL_SKY / (len(grid) + 1)


def coverage_map(centers, radius, nside=NSIDE):
    """Number of circles of ``radius`` around ``centers`` containing each pixel.

    Returns
    -------
    counts : numpy.ndarray
        One entry per HEALPix pixel, ring ordered.
    hpx : astropy_healpix.HEALPix
        The grid the counts are on.
    """
    hpx = HEALPix(nside=nside, order="ring", frame=ICRS())
    pixels = hpx.healpix_to_skycoord(np.arange(hpx.npix))
    ipix, _, _, _ = centers.search_around_sky(pixels, radius)
    return np.bincount(ipix, minlength=hpx.npix), hpx


def summarize(name, centers, radius, nside=NSIDE):
    """One line of the printed summary, and the numbers behind it."""
    counts, hpx = coverage_map(centers, radius, nside)
    pixel_area = hpx.pixel_area.to_value(u.deg**2)
    return {
        "name": name,
        "pointings": len(centers),
        "covering_radius": covering_radius(centers),
        "covers": covering_radius(centers) <= radius,
        "uncovered": (counts == 0).sum() * pixel_area * u.deg**2,
        "mean_depth": counts.mean(),
    }


# -----------------------------------------------------------------------------
# Fields on a globe
# -----------------------------------------------------------------------------
def outline(fov, points_per_side=30):
    """Polygon along the edges of a rectangular field of view.

    The edges are sampled densely so that they stay curved on the globe, and so
    that the part of a field behind the globe drops out of the drawing instead
    of being joined across it.

    Parameters
    ----------
    fov : regions.RectangleSkyRegion
        Field of view in the standard orientation, centred on (0, 0).

    Returns
    -------
    regions.PolygonSkyRegion
    """
    x = 0.5 * fov.width.to_value(u.deg)
    y = 0.5 * fov.height.to_value(u.deg)
    t = np.linspace(-1, 1, points_per_side, endpoint=False)
    lon = np.concatenate([x * t, np.full_like(t, x), -x * t, np.full_like(t, -x)])
    lat = np.concatenate([np.full_like(t, -y), y * t, np.full_like(t, y), -y * t])
    return PolygonSkyRegion(SkyCoord(lon * u.deg, lat * u.deg))


def new_globe(figsize=(4.0, 4.0)):
    """Figure and globe axes without ticks, as in the UVEX sky-grid figure."""
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(projection="astro globe")
    for coord in ax.coords:
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
    fig.tight_layout()
    return fig, ax


def draw_fields(ax, fov, centers, roll):
    """Add the outline of the field of view at every centre.

    Returns
    -------
    list of matplotlib.artist.Artist
        The lines added, so that an animation can remove them.
    """
    transform = ax.get_transform("world")
    artists = []
    for region in footprint(outline(fov), centers, roll):
        ra = region.vertices.ra.deg
        dec = region.vertices.dec.deg
        (line,) = ax.plot(
            np.append(ra, ra[0]),
            np.append(dec, dec[0]),
            color=FIELD_EDGE,
            linewidth=0.3,
            transform=transform,
        )
        artists.append(line)
    return artists


def plot_fields(fov, centers, roll=ROLL):
    """The field of view at every pointing of a grid, on a globe."""
    fig, ax = new_globe()
    draw_fields(ax, fov, centers, roll)
    return fig


def save_gif(fov, centers, path, rolls=GIF_ROLLS):
    """Animate the fields over ``rolls``."""
    fig, ax = new_globe()
    current = []

    def update(roll):
        removed = list(current)
        for artist in removed:
            artist.remove()
        current[:] = draw_fields(ax, fov, centers, roll)
        return removed + current

    FuncAnimation(fig, update, frames=rolls, blit=True, interval=50).save(path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Coverage over the whole sky
# -----------------------------------------------------------------------------
def plot_coverage(centers, radius):
    """How many inscribed circles contain each position on the sky."""
    counts, _ = coverage_map(centers, radius)
    ticks = np.arange(counts.min(), counts.max() + 1)
    # The palette starts at zero, so drop its first entry when nothing is bare.
    colors = COVERAGE_COLORS[counts.min() : counts.max() + 1]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.append(ticks, ticks[-1] + 1) - 0.5, cmap.N)

    fig = plt.figure(figsize=(6.4, 3.4), facecolor="white")
    ax = fig.add_subplot(projection="astro degrees mollweide")
    image = ax.imshow_hpx(
        (counts.astype(float), "ICRS"), nested=False, cmap=cmap, norm=norm
    )
    ax.plot_coord(centers, ".", color=CENTER_COLOR, markersize=2, markeredgewidth=0)
    ax.grid(color="#999999", linewidth=0.3)
    for coord in ax.coords:
        coord.set_ticklabel(size=7)

    cbar = fig.colorbar(
        image, ax=ax, orientation="horizontal", shrink=0.6, pad=0.08, ticks=ticks
    )
    # The inscribed radius is the whole criterion, so name it instead of
    # making the reader decode "inscribed circles containing the position".
    cbar.set_label(f"Pointings within {radius:.2f}", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def save(fig, outdir, stem):
    """Write one panel as its own PDF and PNG."""
    for suffix in ("pdf", "png"):
        fig.savefig(outdir / f"{stem}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--outdir", type=Path, default=Path("paper_outputs"), help="output directory"
    )
    parser.add_argument(
        "--skygrid",
        action="append",
        help="draw one of the mission's own grids instead, repeatable",
    )
    parser.add_argument(
        "--gif", action="store_true", help="also animate the fields over roll"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Imported here: building the mission reads its data files.
    from m4opt.missions import ultrasat

    fov = ultrasat.fov
    radius = inscribed_radius(fov)

    if args.skygrid is None:
        grids = {"geodesic": smallest_covering_grid(radius)}
        # The runs read their grid from the mission, so the grid derived here
        # has to be the mission's. M4OPT defines "geodesic" for ULTRASAT only
        # after 4.0.1; until the pin moves, there is nothing to compare with.
        mission_grid = ultrasat.skygrid.get("geodesic")
        if mission_grid is None:
            print(
                f"m4opt {version('m4opt')} defines no 'geodesic' grid for "
                f"ultrasat, only {list(ultrasat.skygrid)}: not compared\n"
            )
        else:
            agrees = len(mission_grid) == len(grids["geodesic"]) and np.allclose(
                unit_vectors(mission_grid), unit_vectors(grids["geodesic"])
            )
            if not agrees:
                raise SystemExit(
                    "the derived grid is not the one the mission would use: "
                    f"{len(grids['geodesic'])} pointings here against "
                    f"{len(mission_grid)} in m4opt {version('m4opt')}"
                )
            print(
                f"agrees with ultrasat.skygrid['geodesic'] of m4opt {version('m4opt')}\n"
            )
    else:
        grids = {
            name: value() if callable(value) else value
            for name, value in ultrasat.skygrid.items()
            if name in args.skygrid
        }
        if not grids:
            raise SystemExit(
                f"no such grid: {args.skygrid}; the mission has {list(ultrasat.skygrid)}"
            )

    print(f"field of view    {fov.width:.2f} x {fov.height:.2f}")
    print(f"inscribed radius {radius:.3f}\n")
    header = (
        "grid",
        "fields",
        "radius (deg)",
        "covers",
        "uncovered (deg2)",
        "mean depth",
    )
    print(f"{header[0]:14}" + "".join(f"{h:>18}" for h in header[1:]))
    for name, centers in grids.items():
        s = summarize(name, centers, radius)
        print(
            f"{name:14}{s['pointings']:18d}"
            f"{s['covering_radius'].to_value(u.deg):18.3f}"
            f"{'yes' if s['covers'] else 'NO':>18}"
            f"{s['uncovered'].to_value(u.deg**2):18.0f}"
            f"{s['mean_depth']:18.2f}"
        )

    # Its own directory: two panels per grid, and more if --gif is given.
    outdir = args.outdir / "skygrid"
    outdir.mkdir(parents=True, exist_ok=True)
    customize_style()
    for name, centers in grids.items():
        save(plot_fields(fov, centers), outdir, f"{name}_fields")
        save(plot_coverage(centers, radius), outdir, f"{name}_coverage")
        if args.gif:
            save_gif(fov, centers, outdir / f"{name}_fields.gif")
    print(f"\nwrote to {outdir}/")


if __name__ == "__main__":
    main()
