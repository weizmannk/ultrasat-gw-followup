"""Expected kilonova yields of one mission, from its post-processed table.

Everything the paper quotes is computed here; the rendering lives in
:mod:`ultrasat_gw_followup.workflow.paper_results`.

Conventions
-----------
An event is *selected* when the scheduler objective reaches the cutoff stored
in the table. The *detected* count is the sum of the known-position detection
probabilities over the selected events.

Counts are expected numbers per year at the median merger rate,

    lambda = R_CBC / R_sim * N

where ``R_CBC`` is the median rate of the table's GWTC-5.0 population model
and ``R_sim`` the effective rate of the simulation. Both are read from the
table metadata, so a sample cannot be weighted with another model's rate.

Percentiles of the selected events are unweighted; percentiles of the
detections are weighted by the detection probability. Both use the
``inverted_cdf`` convention, the one the area-distance figures draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from astropy.table import QTable

from ultrasat_gw_followup.probability.rate import DEFAULT_POPULATION, MERGER_RATE

#: Observing sub-periods the paper covers.
RUNS = ("O5a", "O5b", "O5c")

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

#: Bins of the area and distance histograms.
AREA_BINS = (0, 100, 500, 1000, 2000, 5000, np.inf)  # deg^2
AREA_LABELS = ("<100", "100–500", "500–1k", "1k–2k", "2k–5k", ">5k")
DISTANCE_BINS = (0, 200, 300, 400, 500, 700, np.inf)  # Mpc
DISTANCE_LABELS = ("<200", "200–300", "300–400", "400–500", "500–700", ">700")

#: Network SNR thresholds of the robustness test.
SNR_THRESHOLDS = (8, 10, 12)


@dataclass
class Results:
    """Scheduling results of one mission, restricted to O5 and sorted by event."""

    name: str
    path: Path
    table: QTable
    population: str  # GWTC-5.0 population model of the sample
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


# -----------------------------------------------------------------------------
# Reading
# -----------------------------------------------------------------------------
def load_results(path, name) -> Results:
    """Read one post-processed result table.

    Parameters
    ----------
    path : str or pathlib.Path
        Table written by ``ugf-postprocess``.
    name : str
        Label used in messages and figure legends.

    Returns
    -------
    Results

    Raises
    ------
    SystemExit
        If the table declares a population model with no known merger rate.
    """
    table = QTable.read(path)
    population = table.meta.get("population_model", DEFAULT_POPULATION)
    if population not in MERGER_RATE:
        raise SystemExit(f"{path}: unknown population model {population!r}")
    merger_rate = MERGER_RATE[population]["mid"]
    rates = {
        run: rate.to_value("Gpc-3 yr-1")
        for run, rate in table.meta["effective_rate"].items()
    }
    table = table[np.isin(np.asarray(table["run"]).astype(str), RUNS)]

    # Sorting on a common key lets two missions be compared event by event.
    key = np.array([f"{r}:{e}" for r, e in zip(table["run"], table["coinc_event_id"])])
    order = np.argsort(key)
    table, key = table[order], key[order]
    run = np.asarray(table["run"]).astype(str)

    return Results(
        name=name,
        path=Path(path),
        table=table,
        population=population,
        key=key,
        run=run,
        weight=np.array([merger_rate / rates[r] for r in run]),
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
    """Network SNR of each event, in the order of ``results``.

    Taken from the ``network_snr`` column when postprocess wrote one. Older
    tables have none -- their ``snr`` column holds the kilonova threshold,
    which postprocess wrote over the GW value -- so it is matched back from
    the observing-scenarios table on ``(run, coinc_event_id)``.

    Raises
    ------
    SystemExit
        If the fallback table is missing some of the events.
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
    """Stop unless the missions are comparable.

    They must hold the same events, come from the same population model, and
    agree on every setting of :data:`MATCHED`. Otherwise their ratio measures
    the difference in assumptions rather than in the missions.

    Raises
    ------
    SystemExit
        On the first difference found.
    """
    reference = missions[0]
    for m in missions[1:]:
        if not np.array_equal(m.key, reference.key):
            raise SystemExit(f"{m.name} and {reference.name} hold different events")
        if m.population != reference.population:
            raise SystemExit(
                f"{m.name} is {m.population} but {reference.name} is "
                f"{reference.population}; the merger-rate priors differ"
            )

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


def setting(results: Results, column) -> str:
    """The single value a scheduler setting takes across a result table.

    Raises
    ------
    SystemExit
        If the table holds more than one, which would make a caption quoting
        it wrong for part of the rows.
    """
    values = {str(value) for value in results.table[column]}
    if len(values) != 1:
        raise SystemExit(
            f"{results.path}: several values of {column}: {sorted(values)}"
        )
    return values.pop()


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
    """``{run: (selected, detected)}`` per sub-period, plus ``"O5"``."""
    out = {r: (m.n_selected(m.run == r), m.n_detected(m.run == r)) for r in RUNS}
    out["O5"] = (m.n_selected(), m.n_detected())
    return out


def gw_counts(m: Results) -> dict:
    """Expected BNS and NSBH GW detections per sub-period, plus ``"O5"``."""
    out = {r: m.weight[m.run == r].sum() for r in RUNS}
    out["O5"] = m.weight.sum()
    return out


def selection_percentiles(m: Results, run: str) -> dict:
    """90th percentiles shown in the boxes of the area-distance figures."""
    in_run = m.run == run
    chosen = in_run & m.selected
    return {
        "DistAll": np.quantile(m.distance[in_run], 0.9, method="inverted_cdf"),
        "DistTrig": np.quantile(m.distance[chosen], 0.9, method="inverted_cdf"),
        "DistDet": weighted_quantile(m.distance[chosen], m.pdet[chosen], 0.9),
        "AreaTrig": np.quantile(m.area[chosen], 0.9, method="inverted_cdf"),
        "AreaDet": weighted_quantile(m.area[chosen], m.pdet[chosen], 0.9),
    }


def gw_sample_medians(m: Results) -> dict:
    """``{run: (median area, median distance)}`` of the BNS and NSBH sample."""
    return {
        r: (np.median(m.area[m.run == r]), np.median(m.distance[m.run == r]))
        for r in RUNS
    }


def binned(m: Results, values, edges) -> list:
    """Detected counts over O5 in each bin of ``values``."""
    return [m.n_detected((values >= lo) & (values < hi)) for lo, hi in pairwise(edges)]


def solver_summary(m: Results) -> dict:
    """How far the solver went: time limit, crashes, and what it left undecided."""
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
    """Events one mission selects and the other does not, and the best of both."""
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
    weight = missions[0].weight
    return {
        threshold: {"GW": weight[snr >= threshold].sum() / weight.sum()}
        | {m.name: m.n_detected(snr >= threshold) / m.n_detected() for m in missions}
        for threshold in SNR_THRESHOLDS
    }
