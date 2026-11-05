"""Numbers, table and figure of the paper, from the final result tables.

Every value the paper quotes is computed from the three post-processed result
tables and the observing-scenarios event table. Re-running a mission and then
this script is enough to bring the paper up to date; nothing is typed by hand.

The statistics themselves live in
:mod:`ultrasat_gw_followup.analysis.mission_yields`, which also documents the
conventions; this module only turns them into LaTeX, a figure and a report.

Usage
-----
::

    ugf-paper-results                              # the PixelPop set
    ugf-paper-results --results-dir results/fullpop

One flag switches population model: the four input tables default to their own
names inside ``--results-dir``, and the outputs go to
``paper_outputs/<model>/``, where the model is the one the tables record. So
each set of results is written beside its own, and the two can never be mixed.

Outputs
-------
``tables/numbers-<model>.tex``
    One ``\\newcommand`` per number quoted in the text.
``tables/mission_comparison-<model>.tex``
    ULTRASAT (LCS) and UVEX per sub-period.
``figures/detections_by_area_distance-<model>.pdf`` (and ``.png``)
    Detected kilonovae per bin of localization area and of luminosity
    distance, for both missions.

A report of every quoted number is printed, followed by a check of the
qualitative statements of the text against the data.
"""

from __future__ import annotations

import argparse
import sys
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.patches import Patch

from ultrasat_gw_followup.analysis.mission_yields import (
    AREA_BINS,
    AREA_LABELS,
    DISTANCE_BINS,
    DISTANCE_LABELS,
    RUNS,
    SNR_THRESHOLDS,
    Results,
    binned,
    check_consistency,
    complementarity,
    gw_counts,
    gw_sample_medians,
    load_results,
    network_snr,
    selection_percentiles,
    setting,
    snr_robustness,
    solver_summary,
    yields,
)
from ultrasat_gw_followup.probability.rate import MERGER_RATE

#: Final tables, one directory per GWTC-5.0 population model. The paper is
#: built from the PixelPop set; the FullPop one is kept for comparison.
RESULTS = Path("results")
DEFAULT_RESULTS = RESULTS / "pixelpop"

#: Outputs, likewise one directory per population model.
PAPER_OUTPUTS = Path("paper_outputs")

#: Suffix the macro names give each sub-period.
LETTER = dict(zip(RUNS, "ABC"))

#: Macro prefix and report label of each scheduled mission.
MISSIONS = (("us", "ULTRASAT LCS"), ("as", "ULTRASAT AllSS"), ("uv", "UVEX"))

#: Categorical colours (validated for colour-vision deficiencies).
COLORS = {"ULTRASAT": "#2a78d6", "UVEX": "#eb6834"}


# -----------------------------------------------------------------------------
# Numbers
#
# Each section below returns ``{macro name: (value, format)}``; collect_numbers
# merges them in the order the paper reads and formats them in one pass. The
# format is the precision the paper prints, and the only place it is chosen.
# -----------------------------------------------------------------------------
def collect_numbers(us, al, uv, snr) -> dict:
    """Every number quoted in the text, as LaTeX macro name -> formatted value.

    Parameters
    ----------
    us, al, uv : Results
        ULTRASAT on the LCS grid, ULTRASAT on the all-sky grid, and UVEX.
    snr : numpy.ndarray
        Network SNR of each event, in the order of the tables.

    Returns
    -------
    dict
    """
    y = {"us": yields(us), "as": yields(al), "uv": yields(uv)}
    gw = gw_counts(us)
    missions = (("us", us), ("as", al), ("uv", uv))
    values = (
        _per_run(y, gw)
        | _totals(y, gw, missions, us)
        | _percentiles(us)
        | _grids(y, us, al)
        | _detection_origins(us, uv)
        | _both_missions(y, us, uv)
        | _snr_losses(snr, us, uv)
        | _solver(missions)
    )
    return {name: format(value, fmt) for name, (value, fmt) in values.items()}


def _per_run(y, gw) -> dict:
    """Selected and detected counts of each sub-period."""
    out = {}
    for run in RUNS:
        letter = LETTER[run]
        out[f"gw{letter}"] = (gw[run], ".1f")
        for prefix in y:
            out[f"{prefix}Sel{letter}"] = (y[prefix][run][0], ".2f")
            out[f"{prefix}Det{letter}"] = (y[prefix][run][1], ".2f")
        out[f"ratioDet{letter}"] = (y["uv"][run][1] / y["us"][run][1], ".1f")
    return out


def _totals(y, gw, missions, sample) -> dict:
    """O5 totals, detection efficiency, and growth over the sub-periods.

    ``sample`` provides the GW event sample, which the three tables share.
    """
    out = {}
    for prefix, m in missions:
        efficiency = [100 * y[prefix][r][1] / y[prefix][r][0] for r in RUNS]
        out[f"{prefix}SelTot"] = (y[prefix]["O5"][0], ".1f")
        out[f"{prefix}DetTot"] = (y[prefix]["O5"][1], ".2f")
        out[f"{prefix}EffMin"] = (min(efficiency), ".0f")
        out[f"{prefix}EffMax"] = (max(efficiency), ".0f")
        out[f"{prefix}Growth"] = (y[prefix]["O5c"][1] / y[prefix]["O5a"][1], ".2f")
        out[f"{prefix}DetBNS"] = (m.n_detected(m.source_class == "BNS"), ".2f")
        out[f"{prefix}DetNSBH"] = (m.n_detected(m.source_class == "NSBH"), ".2f")
        out[f"{prefix}FieldsMed"] = (np.median(m.num_fields[m.selected]), ".0f")

    out["gwTot"] = (gw["O5"], ".1f")
    out["gwGrowth"] = (gw["O5c"] / gw["O5a"], ".2f")
    out["gwGrowthAB"] = (100 * (gw["O5b"] / gw["O5a"] - 1), ".0f")
    out["usGrowthAB"] = (100 * (y["us"]["O5b"][1] / y["us"]["O5a"][1] - 1), ".0f")
    out["uvGrowthAB"] = (100 * (y["uv"]["O5b"][1] / y["uv"]["O5a"][1] - 1), ".0f")
    out["usGrowthBC"] = (100 * (y["us"]["O5c"][1] / y["us"]["O5b"][1] - 1), ".0f")
    out["usDetFracGW"] = (100 * y["us"]["O5"][1] / gw["O5"], ".0f")
    out["ratioDetTot"] = (y["uv"]["O5"][1] / y["us"]["O5"][1], ".1f")

    for run, (area, distance) in gw_sample_medians(sample).items():
        out[f"gwAreaMed{LETTER[run]}"] = (area, ".0f")
        out[f"gwDistMed{LETTER[run]}"] = (distance, ".0f")
    return out


def _percentiles(us) -> dict:
    """90th percentiles written in the boxes of the LCS area-distance figures."""
    return {
        f"us{key}{LETTER[run]}": (value, ".0f")
        for run in RUNS
        for key, value in selection_percentiles(us, run).items()
    }


def _grids(y, us, al) -> dict:
    """The two sky grids: how the O5a selections overlap, and the spread."""
    o5a = us.run == "O5a"

    def spread(index):
        """Largest relative difference between the grids, over the sub-periods."""
        return 100 * max(abs(y["as"][r][index] / y["us"][r][index] - 1) for r in RUNS)

    return {
        "gridNlcsA": (np.sum(us.selected & o5a), "d"),
        "gridNallA": (np.sum(al.selected & o5a), "d"),
        "gridNcommonA": (np.sum(us.selected & al.selected & o5a), "d"),
        "gridNlcsOnlyA": (np.sum(us.selected & ~al.selected & o5a), "d"),
        "gridNallOnlyA": (np.sum(al.selected & ~us.selected & o5a), "d"),
        "gridSelDiff": (spread(0), ".0f"),
        "gridDetDiff": (spread(1), ".0f"),
        "gridDetDiffTot": (100 * abs(y["as"]["O5"][1] / y["us"]["O5"][1] - 1), ".0f"),
    }


def _detection_origins(us, uv) -> dict:
    """Which localization areas and which distances the detections come from."""
    weight = us.weight
    small, large = us.area < 500, us.area >= 2000
    near, far = us.distance < 200, us.distance >= 700

    def gw_fraction(mask):
        """Share of the GW events of a bin whose kilonova ULTRASAT detects."""
        return 100 * us.n_detected(mask) / weight[mask].sum()

    return {
        "usDetSmallArea": (us.n_detected(small), ".1f"),
        "uvDetSmallArea": (uv.n_detected(small), ".1f"),
        "usDetLargeArea": (us.n_detected(large), ".2f"),
        "uvDetLargeArea": (uv.n_detected(large), ".2f"),
        "gwLargeAreaPct": (100 * weight[large].sum() / weight.sum(), ".0f"),
        "usDetLargeAreaPct": (100 * us.n_detected(large) / weight[large].sum(), ".1f"),
        "usDetSmallAreaPct": (100 * us.n_detected(small) / us.n_detected(), ".0f"),
        "usDetNear": (us.n_detected(near), ".1f"),
        "uvDetNear": (uv.n_detected(near), ".1f"),
        "usDetFar": (us.n_detected(far), ".1f"),
        "uvDetFar": (uv.n_detected(far), ".1f"),
        "usFracAreaSmallest": (gw_fraction(us.area < 100), ".0f"),
        "usFracAreaMid": (gw_fraction((us.area >= 1000) & (us.area < 2000)), ".0f"),
        "usFracDistNearest": (gw_fraction(us.distance < 200), ".0f"),
        "usFracDistMid": (
            gw_fraction((us.distance >= 500) & (us.distance < 700)),
            ".0f",
        ),
    }


def _both_missions(y, us, uv) -> dict:
    """What only ULTRASAT selects, and the yield of the two missions together."""
    c = complementarity(us, uv)
    return {
        "usOnlySel": (c["us_only_sel"], ".1f"),
        "usOnlyDet": (c["us_only_det"], ".1f"),
        "usOnlyArea": (c["us_only_area"], ".0f"),
        "bestOfBoth": (c["best_of_both"], ".1f"),
        "bestOfBothGainPct": (100 * (c["best_of_both"] / y["uv"]["O5"][1] - 1), ".0f"),
    }


def _snr_losses(snr, us, uv) -> dict:
    """Yield lost when the GW detection threshold is raised above the nominal 8."""
    robust = snr_robustness(snr, us, uv)
    out = {}
    for threshold in SNR_THRESHOLDS[1:]:
        kept = robust[threshold]
        out[f"snrGwLoss{threshold}"] = (100 * (1 - kept["GW"]), ".0f")
        out[f"snrUsLoss{threshold}"] = (100 * (1 - kept["ULTRASAT"]), ".0f")
        out[f"snrUvLoss{threshold}"] = (100 * (1 - kept["UVEX"]), ".0f")
    return out


def _solver(missions) -> dict:
    """How far the solver got on each mission."""
    out = {}
    for prefix, m in missions:
        s = solver_summary(m)
        out[f"{prefix}LimitPct"] = (s["limited_pct"], ".0f")
        out[f"{prefix}LimitSharePct"] = (s["limited_share_pct"], ".0f")
        out[f"{prefix}CrashPct"] = (s["crashed_pct"], ".1f")
        out[f"{prefix}GapPct"] = (s["median_gap_pct"], ".0f")
        out[f"{prefix}Undecided"] = (s["undecided"], ".1f")
    return out


# -----------------------------------------------------------------------------
# LaTeX
# -----------------------------------------------------------------------------
def write_numbers(path, numbers, sources) -> None:
    """Write one ``\\newcommand`` per number, after the tables they come from."""
    lines = ["% Generated by ugf-paper-results. Do not edit."]
    lines += [f"% {name}: {src}" for name, src in sources.items()]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in numbers.items()]
    Path(path).write_text("\n".join(lines) + "\n")


def write_comparison_table(path, us, uv) -> None:
    """ULTRASAT (LCS) and UVEX per sub-period.

    Every quantity the caption states is read back from the tables, so that it
    cannot drift from the run it describes -- the population model in
    particular, which sets the merger rate and is the only thing that
    distinguishes two otherwise identical-looking tables.
    """
    y_us, y_uv, gw = yields(us), yields(uv), gw_counts(us)
    rows = [
        f"{'O5 total' if r == 'O5' else r} & {gw[r]:.1f} & {y_us[r][0]:.2f} & "
        f"{y_us[r][1]:.2f} & {y_uv[r][0]:.2f} & {y_uv[r][1]:.2f} & "
        f"{y_uv[r][1] / y_us[r][1]:.1f} \\\\"
        for r in (*RUNS, "O5")
    ]
    rows.insert(len(RUNS), r"\midrule")
    body = "\n".join(rows)

    model = us.population
    rate = MERGER_RATE[model]["mid"]
    snr = f"{float(setting(us, 'snr')):g}"
    visits = int(float(setting(us, "visits")))
    delay = u.Quantity(setting(us, "delay")).to(u.min)
    deadline = u.Quantity(setting(us, "deadline")).to(u.hour)
    grid = us.table["skygrid"][0]

    Path(path).write_text(
        rf"""\begin{{table}}[!htb]
\centering
\renewcommand\arraystretch{{1.15}}
\setlength{{\tabcolsep}}{{0.14cm}}
\caption{{ULTRASAT ({grid} grid) and UVEX under identical scheduling
assumptions: the same simulated events, the same kilonova integrated through
each mission's NUV band, SNR\,$={snr}$, {visits} visits, observations between
{delay.value:g}\,min and {deadline.value:g}\,hr after the merger. Events are
drawn from the GWTC-5.0 {model} model, whose median total CBC merger rate is
{rate:g}\,$\mathrm{{Gpc^{{-3}}\,yr^{{-1}}}}$. All values are expected numbers
per year, $\lambda$, at that rate; the 90\% credible intervals are given in
Tables~\ref{{tab:ultrasat-{grid}-selected-detected-1yr}}
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


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def plot_detections_by_area_distance(path, us, uv) -> None:
    """Detected kilonovae over O5 per bin of area and of distance.

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
            "ULTRASAT grows more slowly than the GW sample",
            f["usGrowth"] < f["gwGrowth"],
        ),
        (
            "most ULTRASAT detections come from areas below 500 deg2",
            f["usDetSmallAreaPct"] > 50,
        ),
        (
            "the two missions perform alike above 2000 deg2",
            abs(uv.n_detected(large) / us.n_detected(large) - 1) < 0.25,
        ),
        (
            "UVEX detects more than ULTRASAT",
            f["ratioDetTot"] > 1,
        ),
        (
            "UVEX converts more selected events into detections",
            f["uvEffMin"] > f["usEffMax"],
        ),
        (
            "the yields shrink less than the GW sample at SNR >= 10",
            f["snrUsLoss10"] < f["snrGwLoss10"] and f["snrUvLoss10"] < f["snrGwLoss10"],
        ),
        (
            "the UVEX plans are further from optimal than ULTRASAT's",
            f["uvGapPct"] > f["usGapPct"],
        ),
        (
            "the two grids agree to within 5% in detections over O5",
            f["gridDetDiffTot"] <= 5,
        ),
    ]


def _per_run_line(label, n, name) -> str:
    """One ``label  O5a O5b O5c O5`` line, from the macros ``nameA..nameTot``."""
    cells = [n[f"{name}{letter}"] for letter in "ABC"] + [n[f"{name}Tot"]]
    return f"  {label:28s}" + "".join(f"{cell:>8}" for cell in cells)


def print_report(n: dict, claims: list, us, al, uv) -> None:
    """Every quoted number, grouped as the paper reads, then the checks."""
    header = "Yields per year (lambda)"
    print(f"  {header:28s}" + "".join(f"{h:>8}" for h in (*RUNS, "O5")))
    print(_per_run_line("GW (BNS + NSBH)", n, "gw"))
    for prefix, label in MISSIONS:
        print(_per_run_line(f"{label} selected", n, f"{prefix}Sel"))
        print(_per_run_line(f"{label} detected", n, f"{prefix}Det"))
    print(_per_run_line("UVEX / ULTRASAT detected", n, "ratioDet"))

    print("\n90th percentiles, ULTRASAT LCS    O5a / O5b / O5c")
    for key in ("DistAll", "DistTrig", "DistDet", "AreaTrig", "AreaDet"):
        values = " / ".join(n[f"us{key}{letter}"] for letter in "ABC")
        print(f"  {key:9s} {values}")
    print(
        f"  detected share of the GW events: {n['usFracAreaSmallest']}% (<100 deg2), "
        f"{n['usFracAreaMid']}% (1000-2000 deg2), "
        f"{n['usFracDistNearest']}% (<200 Mpc), {n['usFracDistMid']}% (500-700 Mpc)"
    )

    print("\nULTRASAT LCS")
    print(
        f"  detected over O5: {n['usDetTot']} "
        f"({n['usDetBNS']} BNS, {n['usDetNSBH']} NSBH); "
        f"detected/selected {n['usEffMin']}-{n['usEffMax']}%; "
        f"{n['usDetFracGW']}% of the GW events"
    )
    print(
        f"  growth O5a->O5c: GW x{n['gwGrowth']}, "
        f"ULTRASAT x{n['usGrowth']}, UVEX x{n['uvGrowth']}"
    )
    print(
        f"  O5a->O5b: GW +{n['gwGrowthAB']}%, ULTRASAT +{n['usGrowthAB']}%, "
        f"UVEX +{n['uvGrowthAB']}%; O5b->O5c ULTRASAT +{n['usGrowthBC']}%"
    )
    print(
        f"  GW sample medians, area (deg2): "
        f"{n['gwAreaMedA']} / {n['gwAreaMedB']} / {n['gwAreaMedC']}; "
        f"distance (Mpc): "
        f"{n['gwDistMedA']} / {n['gwDistMedB']} / {n['gwDistMedC']}"
    )

    print("\nSky grids")
    print(
        f"  O5a selected: LCS {n['gridNlcsA']}, AllSS {n['gridNallA']}, "
        f"common {n['gridNcommonA']}, LCS only {n['gridNlcsOnlyA']}, "
        f"AllSS only {n['gridNallOnlyA']}"
    )
    print(
        f"  largest difference per sub-period: selected {n['gridSelDiff']}%, "
        f"detected {n['gridDetDiff']}%; over O5 detected {n['gridDetDiffTot']}%"
    )

    print("\nULTRASAT and UVEX")
    print(
        f"  detected/selected: ULTRASAT {n['usEffMin']}-{n['usEffMax']}%, "
        f"UVEX {n['uvEffMin']}-{n['uvEffMax']}%; "
        f"median fields per plan {n['usFieldsMed']} vs {n['uvFieldsMed']}"
    )
    print(
        f"  areas < 500 deg2: {n['usDetSmallArea']} vs {n['uvDetSmallArea']}; "
        f"> 2000 deg2 ({n['gwLargeAreaPct']}% of the GW events): "
        f"{n['usDetLargeArea']} vs {n['uvDetLargeArea']}"
    )
    print(
        f"  distances < 200 Mpc: {n['usDetNear']} vs {n['uvDetNear']}; "
        f"> 700 Mpc: {n['usDetFar']} vs {n['uvDetFar']}"
    )
    print(
        f"  ULTRASAT only: {n['usOnlySel']} selected, {n['usOnlyDet']} detected, "
        f"median area {n['usOnlyArea']} deg2; best of both {n['bestOfBoth']} "
        f"(+{n['bestOfBothGainPct']}% over UVEX)"
    )

    print("\nLimitations")
    for threshold in SNR_THRESHOLDS[1:]:
        print(
            f"  SNR >= {threshold}: GW -{n[f'snrGwLoss{threshold}']}%, "
            f"ULTRASAT -{n[f'snrUsLoss{threshold}']}%, "
            f"UVEX -{n[f'snrUvLoss{threshold}']}%"
        )
    for prefix, label in MISSIONS:
        print(
            f"  {label:15s} time limit {n[prefix + 'LimitPct']}% of the events "
            f"({n[prefix + 'LimitSharePct']}% of the detections), "
            f"crashed {n[prefix + 'CrashPct']}%, "
            f"median gap {n[prefix + 'GapPct']}%, "
            f"undecided {n[prefix + 'Undecided']}/yr"
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
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="directory holding the four tables of one population model",
    )
    parser.add_argument("--lcs", help="ULTRASAT result table, LCS grid")
    parser.add_argument("--allsky", help="ULTRASAT result table, AllSS grid")
    parser.add_argument("--uvex", help="UVEX result table")
    parser.add_argument(
        "--scenarios",
        help="observing-scenarios table, read only when the result tables "
        "carry no network_snr column",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="output directory; defaults to paper_outputs/<population model>",
    )

    args = parser.parse_args(argv)
    # Each table defaults to its own name inside --results-dir, so switching
    # population model is one flag rather than four paths.
    for option, name in (
        ("lcs", "ultrasat_lcs"),
        ("allsky", "ultrasat_allsky"),
        ("uvex", "uvex"),
        ("scenarios", "observing-scenarios"),
    ):
        if getattr(args, option) is None:
            setattr(args, option, args.results_dir / f"{name}.ecsv")
    return args


def output_paths(outdir, model) -> dict:
    """Where each output goes, split the way the paper's LaTeX includes them.

    Parameters
    ----------
    outdir : pathlib.Path
        Root directory, usually ``paper_outputs/<model>``.
    model : str
        Population model of the tables, which names every file.

    Returns
    -------
    dict
    """
    return {
        "numbers": outdir / "tables" / f"numbers-{model}.tex",
        "comparison": outdir / "tables" / f"mission_comparison-{model}.tex",
        "figure": outdir / "figures" / f"detections_by_area_distance-{model}.pdf",
    }


def run(argv=None) -> dict:
    """Run the full analysis; return the numbers so a notebook can use them."""
    args = parse_args(argv)
    us = load_results(args.lcs, "ULTRASAT")
    al = load_results(args.allsky, "ULTRASAT AllSS")
    uv = load_results(args.uvex, "UVEX")
    check_consistency(us, al, uv)
    snr = network_snr(us, args.scenarios)

    # The model comes from the tables, not from the directory they sit in, so
    # the outputs are named after what was actually weighted.
    model = us.population
    out = args.outdir or PAPER_OUTPUTS / model
    paths = output_paths(out, model)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    numbers = collect_numbers(us, al, uv, snr)
    write_numbers(
        paths["numbers"],
        numbers,
        {
            "ULTRASAT LCS": args.lcs,
            "ULTRASAT AllSS": args.allsky,
            "UVEX": args.uvex,
            "Scenarios": args.scenarios,
        },
    )
    write_comparison_table(paths["comparison"], us, uv)
    plot_detections_by_area_distance(paths["figure"], us, uv)

    print_report(numbers, check_claims(numbers, us, uv), us, al, uv)
    print(f"\n{len(numbers)} macros, from the GWTC-5.0 {model} model:")
    for path in paths.values():
        print(f"  {path}")
    return numbers


def main(argv=None) -> int:
    """Entry point of ``ugf-paper-results``."""
    run(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
