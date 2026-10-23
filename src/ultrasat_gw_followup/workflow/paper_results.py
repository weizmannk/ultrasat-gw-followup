"""Numbers, table and figure of the ULTRASAT paper, from the final result tables.

Every value quoted in Sections 4 and 5 of the paper is computed here, from the
three post-processed result tables and the observing-scenarios event table.
Re-running a mission and then this script is enough to bring the paper up to
date; nothing is typed by hand.

Usage
-----
::

    python -m ultrasat_gw_followup.workflow.paper_results
    ugf-paper-results --lcs output/ultrasat_lcs/ultrasat_lcs.ecsv

With no argument it reads the three tables of ``results/`` and writes to
``paper_outputs/``. The same entry point can be called from a notebook::

    from ultrasat_gw_followup.workflow.paper_results import main
    numbers = main([...same arguments...])

Outputs (in ``--outdir``)
-------------------------
``numbers.tex``
    One ``\\newcommand`` per number quoted in the text.
``mission_comparison.tex``
    Table 5 of the paper: ULTRASAT (LCS) and UVEX per sub-period.
``detections_by_area_distance.pdf`` (and ``.png``)
    Figure 4 of the paper: detected kilonovae per bin of localization area
    and of luminosity distance, for both missions.

A plain-text report of every quoted number is printed, followed by a check of
the qualitative statements of the text against the data.

Conventions
-----------
* An event is *selected* when the scheduler objective is at least the cutoff
  stored in the table (0.1).
* The *detected* count is the sum of the known-position detection
  probabilities of the selected events.
* Counts are expected numbers per year at the median merger rate,
  ``lambda = R_CBC / R_sim * N``, where ``R_sim`` is the effective rate of the
  simulation stored in the table metadata. They are summed over the three O5
  sub-periods for the O5 totals.
* Percentiles of the selected events are unweighted; percentiles of the
  detections are weighted by the detection probability. Both use the
  ``inverted_cdf`` convention of the area--distance figures.

Note
----
``postprocess.py`` copies the scheduler arguments into the result tables, and
its ``snr`` argument (the kilonova detection threshold, 5) overwrites the
network SNR of the GW event. Tables written by the current postprocess keep
the network SNR in a ``network_snr`` column; older ones do not, and for those
it is read from the observing-scenarios table, matched on
``(run, coinc_event_id)``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from matplotlib.patches import Patch

#: Default location of the three final tables, relative to the repository.
RESULTS = Path("results")

RUNS = ("O5a", "O5b", "O5c")
R_CBC = 100.0  # Gpc^-3 yr^-1, GWTC-5.0 median total CBC rate

#: Scheduler settings that must be identical for the comparison to be
#: controlled. ``absmag_mean`` is deliberately absent: it is the same source
#: evaluated in two different bands.
MATCHED = (
    "delay",
    "deadline",
    "visits",
    "exptime_min",
    "exptime_max",
    "nside",
    "cutoff",
    "absmag_stdev",
    "bandpass",
)

#: Bins of Figure 4.
AREA_BINS = (0, 100, 500, 1000, 2000, 5000, np.inf)  # deg^2
AREA_LABELS = ("<100", "100–500", "500–1k", "1k–2k", "2k–5k", ">5k")
DISTANCE_BINS = (0, 200, 300, 400, 500, 700, np.inf)  # Mpc
DISTANCE_LABELS = ("<200", "200–300", "300–400", "400–500", "500–700", ">700")

#: Network SNR thresholds of the robustness test (Section 5.3).
SNR_THRESHOLDS = (8, 10, 12)

#: Categorical colours (validated for colour-vision deficiencies).
COLORS = {"ULTRASAT": "#2a78d6", "UVEX": "#eb6834"}


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
@dataclass
class Results:
    """Scheduling results of one mission, restricted to O5 and sorted by event."""

    name: str
    path: Path
    table: QTable
    key: np.ndarray  # "run:coinc_event_id"
    run: np.ndarray
    weight: np.ndarray  # expected events per year represented by each event
    objective: np.ndarray
    best_bound: np.ndarray
    pdet: np.ndarray
    status: np.ndarray
    distance: np.ndarray  # true luminosity distance (Mpc)
    area: np.ndarray  # 90% credible area (deg^2)
    source_class: np.ndarray
    num_fields: np.ndarray
    cutoff: float

    @property
    def selected(self) -> np.ndarray:
        return self.objective >= self.cutoff

    def n_selected(self, mask=True) -> float:
        """Expected number of selected events per year (summed over the mask)."""
        return self.weight[self.selected & mask].sum()

    def n_detected(self, mask=True) -> float:
        """Expected number of detected kilonovae per year (summed over the mask)."""
        return (self.weight * self.pdet)[self.selected & mask].sum()


def _column(table, name, dtype=float):
    return np.asarray(table[name], dtype)


def load_results(path, name) -> Results:
    """Read one post-processed result table."""
    table = QTable.read(path)
    rates = {
        run: rate.to_value("Gpc-3 yr-1")
        for run, rate in table.meta["effective_rate"].items()
    }
    table = table[np.isin(np.asarray(table["run"]).astype(str), RUNS)]
    key = np.array([f"{r}:{e}" for r, e in zip(table["run"], table["coinc_event_id"])])
    order = np.argsort(key)
    table, key = table[order], key[order]
    run = np.asarray(table["run"]).astype(str)
    return Results(
        name=name,
        path=Path(path),
        table=table,
        key=key,
        run=run,
        weight=np.array([R_CBC / rates[r] for r in run]),
        objective=np.nan_to_num(_column(table, "objective_value")),
        best_bound=np.nan_to_num(_column(table, "best_bound")),
        pdet=np.nan_to_num(_column(table, "detection_probability_known_position")),
        status=np.asarray(table["solution_status"]).astype(str),
        distance=_column(table, "distance"),
        area=_column(table, "area(90)"),
        source_class=np.asarray(table["source_class"]).astype(str),
        num_fields=_column(table, "num_fields"),
        cutoff=float(table["cutoff"][0]),
    )


def network_snr(results: Results, scenarios) -> np.ndarray:
    """Network SNR of each event of a result table.

    Taken from the ``network_snr`` column when postprocess wrote one, and
    otherwise from the observing-scenarios table, matched on
    ``(run, coinc_event_id)``.

    Parameters
    ----------
    results : Results
        Mission whose events set the order of the output.
    scenarios : str or pathlib.Path
        Observing-scenarios event table, used as the fallback source.

    Returns
    -------
    numpy.ndarray
        One network SNR per event of ``results``, in its order.
    """
    if "network_snr" in results.table.colnames:
        return _column(results.table, "network_snr")

    table = QTable.read(scenarios)
    snr = {
        f"{r}:{e}": s
        for r, e, s in zip(table["run"], table["coinc_event_id"], _column(table, "snr"))
    }
    missing = [key for key in results.key if key not in snr]
    if missing:
        raise SystemExit(
            f"{len(missing)} events missing from {scenarios}, e.g. {missing[0]}"
        )
    return np.array([snr[key] for key in results.key])


def check_consistency(*missions: Results) -> None:
    """Stop if the missions hold different events or differ in a matched setting."""
    reference = missions[0]
    for m in missions[1:]:
        if not np.array_equal(m.key, reference.key):
            raise SystemExit(f"{m.name} and {reference.name} hold different events")
    problems = []
    for column in MATCHED:
        values = {
            m.name: tuple(sorted(set(np.asarray(m.table[column]).astype(str))))
            for m in missions
            if column in m.table.colnames
        }
        if len(set(values.values())) > 1:
            problems.append(f"  {column}: {values}")
    if problems:
        raise SystemExit("Settings differ between missions:\n" + "\n".join(problems))


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
def weighted_quantile(values, weights, q) -> float:
    """Weighted quantile with the ``inverted_cdf`` convention of numpy."""
    order = np.argsort(values)
    cumulative = np.cumsum(np.asarray(weights, float)[order])
    index = np.searchsorted(cumulative, q * cumulative[-1])
    return float(np.asarray(values)[order][min(index, len(values) - 1)])


def yields(m: Results) -> dict:
    """Selected and detected counts per sub-period and over O5."""
    out = {r: (m.n_selected(m.run == r), m.n_detected(m.run == r)) for r in RUNS}
    out["O5"] = (m.n_selected(), m.n_detected())
    return out


def gw_counts(m: Results) -> dict:
    """Expected BNS and NSBH GW detections per sub-period and over O5."""
    out = {r: m.weight[m.run == r].sum() for r in RUNS}
    out["O5"] = m.weight.sum()
    return out


def selection_percentiles(m: Results, run: str) -> dict:
    """90th percentiles shown in the boxes of the area--distance figures."""
    r = m.run == run
    s = r & m.selected
    return {
        "DistAll": np.quantile(m.distance[r], 0.9, method="inverted_cdf"),
        "DistTrig": np.quantile(m.distance[s], 0.9, method="inverted_cdf"),
        "DistDet": weighted_quantile(m.distance[s], m.pdet[s], 0.9),
        "AreaTrig": np.quantile(m.area[s], 0.9, method="inverted_cdf"),
        "AreaDet": weighted_quantile(m.area[s], m.pdet[s], 0.9),
    }


def gw_sample_medians(m: Results) -> dict:
    """Median 90% area and distance of the BNS and NSBH sample, per sub-period."""
    return {
        r: (np.median(m.area[m.run == r]), np.median(m.distance[m.run == r]))
        for r in RUNS
    }


def binned(m: Results, values, edges) -> list:
    """Detected counts over O5 in each bin of ``values``."""
    return [m.n_detected((values >= lo) & (values < hi)) for lo, hi in pairwise(edges)]


def solver_summary(m: Results) -> dict:
    """How far the solver went: time limit, crashes, and bounds on what it missed."""
    limited = m.status == "time limit exceeded"
    crashed = np.char.startswith(m.status, "CPLEX Error")
    # Not selected, but CPLEX could not rule out a plan above the cutoff.
    undecided = ~m.selected & (m.best_bound >= m.cutoff)
    sel_limited = m.selected & limited
    gap = (m.best_bound - m.objective)[sel_limited] / m.objective[sel_limited]
    return {
        "limited_pct": 100 * limited.mean(),
        "crashed_pct": 100 * crashed.mean(),
        "limited_share_pct": 100 * m.n_detected(limited) / m.n_detected(),
        "undecided": m.weight[undecided].sum(),
        "median_gap_pct": 100 * np.median(gap) if gap.size else 0.0,
    }


def complementarity(us: Results, uv: Results) -> dict:
    """Events selected by one mission only, and the best of both per event."""
    us_only = us.selected & ~uv.selected
    pdet_us = np.where(us.selected, us.pdet, 0)
    pdet_uv = np.where(uv.selected, uv.pdet, 0)
    return {
        "us_only_sel": us.weight[us_only].sum(),
        "us_only_det": (us.weight * us.pdet)[us_only].sum(),
        "us_only_area": np.median(us.area[us_only]),
        "uv_only_sel": uv.weight[uv.selected & ~us.selected].sum(),
        "best_of_both": (us.weight * np.maximum(pdet_us, pdet_uv)).sum(),
    }


def snr_robustness(snr, *missions: Results) -> dict:
    """Fraction of GW events and of detections kept above each SNR threshold."""
    w = missions[0].weight
    out = {}
    for threshold in SNR_THRESHOLDS:
        keep = snr >= threshold
        out[threshold] = {"GW": w[keep].sum() / w.sum()} | {
            m.name: m.n_detected(keep) / m.n_detected() for m in missions
        }
    return out


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
def collect_numbers(us, al, uv, snr) -> dict:
    """Every number quoted in the text, as LaTeX macro name -> formatted value."""
    y = {"us": yields(us), "as": yields(al), "uv": yields(uv)}
    gw = gw_counts(us)
    letter = dict(zip(RUNS, "ABC"))
    n = {}

    def put(name, value, fmt):
        n[name] = format(value, fmt)

    # Yields per sub-period (Tables 4 and 5).
    for r in RUNS:
        L = letter[r]
        put(f"gw{L}", gw[r], ".1f")
        for p in y:
            put(f"{p}Sel{L}", y[p][r][0], ".2f")
            put(f"{p}Det{L}", y[p][r][1], ".2f")
        put(f"ratioDet{L}", y["uv"][r][1] / y["us"][r][1], ".1f")

    # O5 totals, efficiencies and growth (Sections 4.3 and 5.1).
    for p, m in (("us", us), ("as", al), ("uv", uv)):
        put(f"{p}SelTot", y[p]["O5"][0], ".1f")
        put(f"{p}DetTot", y[p]["O5"][1], ".2f")
        eff = [100 * y[p][r][1] / y[p][r][0] for r in RUNS]
        put(f"{p}EffMin", min(eff), ".0f")
        put(f"{p}EffMax", max(eff), ".0f")
        put(f"{p}Growth", y[p]["O5c"][1] / y[p]["O5a"][1], ".2f")
        put(f"{p}DetBNS", m.n_detected(m.source_class == "BNS"), ".2f")
        put(f"{p}DetNSBH", m.n_detected(m.source_class == "NSBH"), ".2f")
        put(f"{p}FieldsMed", np.median(m.num_fields[m.selected]), ".0f")
    put("gwTot", gw["O5"], ".1f")
    put("gwGrowth", gw["O5c"] / gw["O5a"], ".2f")
    put("gwGrowthAB", 100 * (gw["O5b"] / gw["O5a"] - 1), ".0f")
    put("usGrowthAB", 100 * (y["us"]["O5b"][1] / y["us"]["O5a"][1] - 1), ".0f")
    put("uvGrowthAB", 100 * (y["uv"]["O5b"][1] / y["uv"]["O5a"][1] - 1), ".0f")
    put("usGrowthBC", 100 * (y["us"]["O5c"][1] / y["us"]["O5b"][1] - 1), ".0f")
    put("usDetFracGW", 100 * y["us"]["O5"][1] / gw["O5"], ".0f")
    put("ratioDetTot", y["uv"]["O5"][1] / y["us"]["O5"][1], ".1f")
    for r, (area, dist) in gw_sample_medians(us).items():
        put(f"gwAreaMed{letter[r]}", area, ".0f")
        put(f"gwDistMed{letter[r]}", dist, ".0f")

    # Percentiles of the LCS area--distance figures (Section 4.1).
    for r in RUNS:
        for k, v in selection_percentiles(us, r).items():
            put(f"us{k}{letter[r]}", v, ".0f")

    # Grid comparison (Section 4.4).
    a = us.run == "O5a"
    put("gridNlcsA", np.sum(us.selected & a), "d")
    put("gridNallA", np.sum(al.selected & a), "d")
    put("gridNcommonA", np.sum(us.selected & al.selected & a), "d")
    put("gridNlcsOnlyA", np.sum(us.selected & ~al.selected & a), "d")
    put("gridNallOnlyA", np.sum(al.selected & ~us.selected & a), "d")
    put(
        "gridSelDiff",
        100 * max(abs(y["as"][r][0] / y["us"][r][0] - 1) for r in RUNS),
        ".0f",
    )
    put(
        "gridDetDiff",
        100 * max(abs(y["as"][r][1] / y["us"][r][1] - 1) for r in RUNS),
        ".0f",
    )
    put("gridDetDiffTot", 100 * abs(y["as"]["O5"][1] / y["us"]["O5"][1] - 1), ".0f")

    # Where the detections come from (Section 4.1, Figure 4).
    w = us.weight
    small, large = us.area < 500, us.area >= 2000
    put("usDetSmallArea", us.n_detected(small), ".1f")
    put("uvDetSmallArea", uv.n_detected(small), ".1f")
    put("usDetLargeArea", us.n_detected(large), ".2f")
    put("uvDetLargeArea", uv.n_detected(large), ".2f")
    put("gwLargeAreaPct", 100 * w[large].sum() / w.sum(), ".0f")
    put("usDetLargeAreaPct", 100 * us.n_detected(large) / w[large].sum(), ".1f")
    put("usDetSmallAreaPct", 100 * us.n_detected(small) / us.n_detected(), ".0f")
    near, far = us.distance < 200, us.distance >= 700
    put("usDetNear", us.n_detected(near), ".1f")
    put("uvDetNear", uv.n_detected(near), ".1f")
    put("usDetFar", us.n_detected(far), ".1f")
    put("uvDetFar", uv.n_detected(far), ".1f")

    # Fraction of the GW events of a bin whose kilonova ULTRASAT detects.
    def detected_fraction(mask):
        return 100 * us.n_detected(mask) / w[mask].sum()

    put("usFracAreaSmallest", detected_fraction(us.area < 100), ".0f")
    put("usFracAreaMid", detected_fraction((us.area >= 1000) & (us.area < 2000)), ".0f")
    put("usFracDistNearest", detected_fraction(us.distance < 200), ".0f")
    put(
        "usFracDistMid",
        detected_fraction((us.distance >= 500) & (us.distance < 700)),
        ".0f",
    )

    # Complementarity (Section 5.1).
    c = complementarity(us, uv)
    put("usOnlySel", c["us_only_sel"], ".1f")
    put("usOnlyDet", c["us_only_det"], ".1f")
    put("usOnlyArea", c["us_only_area"], ".0f")
    put("bestOfBoth", c["best_of_both"], ".1f")
    put("bestOfBothGainPct", 100 * (c["best_of_both"] / y["uv"]["O5"][1] - 1), ".0f")

    # Robustness to the GW detection criterion (Section 5.3).
    robust = snr_robustness(snr, us, uv)
    for threshold in SNR_THRESHOLDS[1:]:
        r = robust[threshold]
        put(f"snrGwLoss{threshold}", 100 * (1 - r["GW"]), ".0f")
        put(f"snrUsLoss{threshold}", 100 * (1 - r["ULTRASAT"]), ".0f")
        put(f"snrUvLoss{threshold}", 100 * (1 - r["UVEX"]), ".0f")

    # Solver convergence (Section 5.3).
    for p, m in (("us", us), ("as", al), ("uv", uv)):
        s = solver_summary(m)
        put(f"{p}LimitPct", s["limited_pct"], ".0f")
        put(f"{p}LimitSharePct", s["limited_share_pct"], ".0f")
        put(f"{p}CrashPct", s["crashed_pct"], ".1f")
        put(f"{p}GapPct", s["median_gap_pct"], ".0f")
        put(f"{p}Undecided", s["undecided"], ".1f")
    return n


def write_numbers(path, numbers, sources) -> None:
    lines = ["% Generated by ugf-paper-results. Do not edit."]
    lines += [f"% {name}: {src}" for name, src in sources.items()]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in numbers.items()]
    Path(path).write_text("\n".join(lines) + "\n")


def write_comparison_table(path, us, uv) -> None:
    """Table 5: ULTRASAT (LCS) and UVEX per sub-period."""
    y_us, y_uv, gw = yields(us), yields(uv), gw_counts(us)
    rows = [
        f"{'O5 total' if r == 'O5' else r} & {gw[r]:.1f} & {y_us[r][0]:.2f} & "
        f"{y_us[r][1]:.2f} & {y_uv[r][0]:.2f} & {y_uv[r][1]:.2f} & "
        f"{y_uv[r][1] / y_us[r][1]:.1f} \\\\"
        for r in (*RUNS, "O5")
    ]
    rows.insert(len(RUNS), r"\midrule")
    body = "\n".join(rows)
    Path(path).write_text(
        rf"""\begin{{table}}[!htb]
\centering
\renewcommand\arraystretch{{1.15}}
\setlength{{\tabcolsep}}{{0.14cm}}
\caption{{ULTRASAT (LCS grid) and UVEX under identical scheduling assumptions:
the same simulated events, the same kilonova integrated through each mission's
NUV band, SNR\,$=5$, two visits, observations between 15\,min and 6\,hr after
the merger. All values are expected numbers per year, $\lambda$, at the median
merger rate; the 90\% credible intervals are given in
Tables~\ref{{tab:ultrasat-non-overlap-selected-detected-1yr}}
and~\ref{{tab:uvex-selected-detected-1yr}}. \emph{{GW}}: BNS and NSBH detections.
\emph{{Sel.}}: selected events; \emph{{Det.}}: detected kilonovae.
\emph{{Ratio}}: UVEX over ULTRASAT detected counts. Because both missions follow
the same events, the merger-rate uncertainty cancels in this ratio.}}
\label{{tab:mission-comparison}}
\begin{{tabular}}{{lcccccc}}
\toprule
 & & \multicolumn{{2}}{{c}}{{ULTRASAT}} & \multicolumn{{2}}{{c}}{{UVEX}} & \\
\cmidrule(lr){{3-4}} \cmidrule(lr){{5-6}}
Run & GW & Sel. & Det. & Sel. & Det. & Ratio \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    )


def plot_detections_by_area_distance(path, us, uv) -> None:
    """Figure 4: detected kilonovae over O5 per bin of area and of distance.

    Both panels share the same vertical scale. The number of BNS and NSBH GW
    detections of each bin is written under its label.
    """
    ink, muted, grid = "#2b2b28", "#7a7a74", "#e8e8e4"
    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.edgecolor": muted,
        "axes.linewidth": 0.6,
        "axes.labelcolor": ink,
        "xtick.color": ink,
        "ytick.color": ink,
        "xtick.major.size": 0,
        "ytick.major.size": 2.5,
        "ytick.major.width": 0.6,
        "savefig.bbox": "standard",
    }
    # Start from the matplotlib defaults, so that the style of the calling
    # notebook (font sizes, themes) cannot change the figure of the paper.
    with plt.style.context("default"), mpl.rc_context(style):
        _draw_detections_by_area_distance(path, us, uv, ink, muted, grid)


def _draw_detections_by_area_distance(path, us, uv, ink, muted, grid) -> None:
    panels = (
        ("area", AREA_BINS, AREA_LABELS, r"90% credible area (deg$^2$)"),
        ("distance", DISTANCE_BINS, DISTANCE_LABELS, "Luminosity distance (Mpc)"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.6), sharey=True)
    width = 0.36
    for ax, (key, edges, labels, xlabel), tag in zip(axes, panels, "ab"):
        values = getattr(us, key)
        x = np.arange(len(labels))
        for j, m in enumerate((us, uv)):
            ax.bar(
                x + (j - 0.5) * (width + 0.03),
                binned(m, values, edges),
                width=width,
                color=COLORS[m.name],
                linewidth=0,
                zorder=3,
            )
        for xi, (lo, hi) in zip(x, pairwise(edges)):
            n_gw = us.weight[(values >= lo) & (values < hi)].sum()
            ax.annotate(
                f"{n_gw:.0f} GW",
                (xi, 0),
                xytext=(0, -17),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=6.8,
                color=muted,
                annotation_clip=False,
            )
        ax.set_xticks(x, labels)
        ax.set_xlabel(xlabel, labelpad=15)
        ax.set_ylabel("Detected kilonovae over O5")
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylim(0, 5.6)
        ax.set_yticks(range(6))
        ax.grid(axis="y", color=grid, linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.015, 0.97, f"({tag})", transform=ax.transAxes, va="top", color=ink)
    fig.legend(
        handles=[Patch(color=COLORS[m.name], label=m.name) for m in (us, uv)],
        loc="upper center",
        ncol=2,
        frameon=False,
        handlelength=1.0,
        handleheight=0.8,
        columnspacing=1.6,
        bbox_to_anchor=(0.55, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.0)
    fig.savefig(path)
    fig.savefig(Path(path).with_suffix(".png"), dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------
def check_claims(n: dict, us: Results, uv: Results) -> list:
    """Qualitative statements of the text, each tested against the data."""
    f = {k: float(v) for k, v in n.items()}
    large = us.area >= 2000
    return [
        (
            "Abstract, Sec. 4.3: ULTRASAT grows more slowly than the GW sample",
            f["usGrowth"] < f["gwGrowth"],
        ),
        (
            "Abstract, Sec. 1: most ULTRASAT detections come from areas below 500 deg2",
            f["usDetSmallAreaPct"] > 50,
        ),
        (
            "Abstract, Sec. 5.1: the two missions perform alike above 2000 deg2",
            abs(uv.n_detected(large) / us.n_detected(large) - 1) < 0.25,
        ),
        (
            "Abstract, Sec. 1, 5.1: UVEX detects more than ULTRASAT",
            f["ratioDetTot"] > 1,
        ),
        (
            "Sec. 5.1: UVEX converts more selected events into detections",
            f["uvEffMin"] > f["usEffMax"],
        ),
        (
            "Sec. 5.3: the yields shrink less than the GW sample at SNR >= 10",
            f["snrUsLoss10"] < f["snrGwLoss10"] and f["snrUvLoss10"] < f["snrGwLoss10"],
        ),
        (
            "Sec. 5.3: the UVEX plans are further from optimal than ULTRASAT's",
            f["uvGapPct"] > f["usGapPct"],
        ),
        (
            "Sec. 4.4: the two grids agree to within 5% in detections over O5",
            f["gridDetDiffTot"] <= 5,
        ),
    ]


def print_report(n: dict, claims: list, us, al, uv) -> None:
    """Every quoted number, grouped by section, then the checks of the text."""
    print("Yields per year (lambda)          O5a     O5b     O5c      O5")
    print(
        f"  GW (BNS + NSBH)              {n['gwA']:>7} {n['gwB']:>7} {n['gwC']:>7} {n['gwTot']:>7}"
    )
    for p, label in (("us", "ULTRASAT LCS"), ("as", "ULTRASAT AllSS"), ("uv", "UVEX")):
        print(
            f"  {label:16s} selected   {n[p + 'SelA']:>7} {n[p + 'SelB']:>7} {n[p + 'SelC']:>7} {n[p + 'SelTot']:>7}"
        )
        print(
            f"  {'':16s} detected   {n[p + 'DetA']:>7} {n[p + 'DetB']:>7} {n[p + 'DetC']:>7} {n[p + 'DetTot']:>7}"
        )
    print(
        f"  UVEX / ULTRASAT detected     {n['ratioDetA']:>7} {n['ratioDetB']:>7} {n['ratioDetC']:>7} {n['ratioDetTot']:>7}"
    )

    print("\nSec. 4.1  90th percentiles (LCS)   O5a / O5b / O5c")
    for k in ("DistAll", "DistTrig", "DistDet", "AreaTrig", "AreaDet"):
        print(
            f"  {k:9s} {n['us' + k + 'A']} / {n['us' + k + 'B']} / {n['us' + k + 'C']}"
        )
    print(
        f"  detected fraction of GW events: {n['usFracAreaSmallest']}% (<100 deg2), "
        f"{n['usFracAreaMid']}% (1000-2000 deg2), {n['usFracDistNearest']}% (<200 Mpc), "
        f"{n['usFracDistMid']}% (500-700 Mpc)"
    )

    print("\nSec. 4.3  ULTRASAT LCS")
    print(
        f"  detected over O5: {n['usDetTot']} ({n['usDetBNS']} BNS, {n['usDetNSBH']} NSBH); "
        f"detected/selected {n['usEffMin']}-{n['usEffMax']}%; {n['usDetFracGW']}% of GW events"
    )
    print(
        f"  growth O5a->O5c: GW x{n['gwGrowth']}, ULTRASAT x{n['usGrowth']}, UVEX x{n['uvGrowth']}"
    )
    print(
        f"  O5a->O5b: GW +{n['gwGrowthAB']}%, ULTRASAT +{n['usGrowthAB']}%, UVEX +{n['uvGrowthAB']}%;"
        f" O5b->O5c ULTRASAT +{n['usGrowthBC']}%"
    )
    print(
        f"  GW sample medians, area (deg2): {n['gwAreaMedA']} / {n['gwAreaMedB']} / {n['gwAreaMedC']}; "
        f"distance (Mpc): {n['gwDistMedA']} / {n['gwDistMedB']} / {n['gwDistMedC']}"
    )

    print("\nSec. 4.4  Grids")
    print(
        f"  O5a selected: LCS {n['gridNlcsA']}, AllSS {n['gridNallA']}, common {n['gridNcommonA']}, "
        f"LCS only {n['gridNlcsOnlyA']}, AllSS only {n['gridNallOnlyA']}"
    )
    print(
        f"  max difference per sub-period: selected {n['gridSelDiff']}%, detected {n['gridDetDiff']}%;"
        f" O5 detected {n['gridDetDiffTot']}%"
    )

    print("\nSec. 5.1  ULTRASAT and UVEX")
    print(
        f"  detected/selected: ULTRASAT {n['usEffMin']}-{n['usEffMax']}%, UVEX {n['uvEffMin']}-{n['uvEffMax']}%;"
        f" median fields per plan {n['usFieldsMed']} vs {n['uvFieldsMed']}"
    )
    print(
        f"  areas < 500 deg2: {n['usDetSmallArea']} vs {n['uvDetSmallArea']};"
        f" > 2000 deg2 ({n['gwLargeAreaPct']}% of GW): {n['usDetLargeArea']} vs {n['uvDetLargeArea']}"
    )
    print(
        f"  distances < 200 Mpc: {n['usDetNear']} vs {n['uvDetNear']}; > 700 Mpc: {n['usDetFar']} vs {n['uvDetFar']}"
    )
    print(
        f"  ULTRASAT only: {n['usOnlySel']} selected, {n['usOnlyDet']} detected, median area {n['usOnlyArea']} deg2;"
        f" best of both {n['bestOfBoth']} (+{n['bestOfBothGainPct']}% over UVEX)"
    )

    print("\nSec. 5.3  Limitations")
    for t in SNR_THRESHOLDS[1:]:
        print(
            f"  SNR >= {t}: GW -{n[f'snrGwLoss{t}']}%, ULTRASAT -{n[f'snrUsLoss{t}']}%, UVEX -{n[f'snrUvLoss{t}']}%"
        )
    for p, label in (("us", "ULTRASAT LCS"), ("as", "ULTRASAT AllSS"), ("uv", "UVEX")):
        print(
            f"  {label:15s} time limit {n[p + 'LimitPct']}% of events ({n[p + 'LimitSharePct']}% of detections),"
            f" crashed {n[p + 'CrashPct']}%, median gap {n[p + 'GapPct']}%, undecided {n[p + 'Undecided']}/yr"
        )

    print("\nQualitative statements in the text:")
    for text, holds in claims:
        print(f"  [{'ok' if holds else 'CHECK THE TEXT'}] {text}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--lcs",
        default=RESULTS / "ultrasat_lcs.ecsv",
        help="ULTRASAT result table, LCS grid",
    )
    parser.add_argument(
        "--allsky",
        default=RESULTS / "ultrasat_allsky.ecsv",
        help="ULTRASAT result table, AllSS grid",
    )
    parser.add_argument(
        "--uvex", default=RESULTS / "uvex.ecsv", help="UVEX result table"
    )
    parser.add_argument(
        "--scenarios",
        default=RESULTS / "observing-scenarios.ecsv",
        help="observing-scenarios table, read only when the result tables "
        "carry no network_snr column",
    )
    parser.add_argument("--outdir", default="paper_outputs", help="output directory")
    return parser.parse_args(argv)


def run(argv=None) -> dict:
    """Run the full analysis; return the numbers so a notebook can use them."""
    args = parse_args(argv)
    us = load_results(args.lcs, "ULTRASAT")
    al = load_results(args.allsky, "ULTRASAT AllSS")
    uv = load_results(args.uvex, "UVEX")
    check_consistency(us, al, uv)
    snr = network_snr(us, args.scenarios)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    numbers = collect_numbers(us, al, uv, snr)
    write_numbers(
        out / "numbers.tex",
        numbers,
        {
            "ULTRASAT LCS": args.lcs,
            "ULTRASAT AllSS": args.allsky,
            "UVEX": args.uvex,
            "Scenarios": args.scenarios,
        },
    )
    write_comparison_table(out / "mission_comparison.tex", us, uv)
    plot_detections_by_area_distance(out / "detections_by_area_distance.pdf", us, uv)

    print_report(numbers, check_claims(numbers, us, uv), us, al, uv)
    print(
        f"\nWrote numbers.tex ({len(numbers)} macros), mission_comparison.tex and "
        f"detections_by_area_distance.pdf to {out}/"
    )
    return numbers


def main(argv=None) -> int:
    """Entry point of ``ugf-paper-results``."""
    run(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
