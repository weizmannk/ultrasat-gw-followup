# Early UV follow-up of gravitational wave events with ULTRASAT

Everything needed to reproduce the paper, from the Zenodo archive to the
numbers, the table and the figure: download, unpack, schedule, post-process,
report.

Three runs are compared, one configuration file each:

| Run | Mission | Sky grid | Kilonova `absmag_mean` | Configuration |
|---|---|---|---|---|
| LCS | ULTRASAT | `non-overlap` | −16.0 | `config/ultrasat_lcs.ini` |
| AllSS | ULTRASAT | `allsky` | −16.0 | `config/ultrasat_allsky.ini` |
| UVEX | UVEX | single grid | −15.6 | `config/uvex.ini` |

Everything else is shared: SNR 5, two visits at least 30 min apart, exposures
from 300 s, observations between 15 min and 6 h after the merger, objective
cutoff 0.1, `nside` 128, runs O5a, O5b and O5c.

## Installation

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[test]"
```

or with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Scheduling needs the full CPLEX engine, not the size-limited runtime that
comes with the `cplex` wheel. Install CPLEX Optimization Studio, then run:

```bash
./scripts/setup-cplex.sh
```

Run it again after any reinstall of the `cplex` wheel. Steps 1, 2, 4 and 5
need no CPLEX; only step 3 does.

## Steps

Every step takes the same `--config`, and every option of the file can be
overridden on the command line. The module form
(`python -m ultrasat_gw_followup.workflow.<step>`) and the installed command
(`ugf-<step>`) are interchangeable.

### 1. Download the observing scenarios

```bash
ugf-download --config config/ultrasat_lcs.ini
```

Fetches `runs.zip` (42 GiB) from Zenodo record
[22555949](https://doi.org/10.5281/zenodo.22555949) into `data/`: the GWTC-5.0
**PixelPop** release. Its FullPop companion is
[22550048](https://doi.org/10.5281/zenodo.22550048); switching between the two
means changing `[download] record` and `[scenarios] pop` together, and the
merger-rate prior follows automatically (see step 5).

The record is pinned, not resolved through the concept DOI: a later version
would change the results. An interrupted download is resumed, a file that is
already complete and verifies is left alone, and the MD5 published by Zenodo
is checked before the archive is accepted. Once is enough — the three runs
share `data/`.

### 2. Unpack it

```bash
ugf-unpacker --config config/ultrasat_lcs.ini
```

Writes `data/observing-scenarios.ecsv` (the BNS and NSBH events of O5a, O5b
and O5c, secondary mass below 3 M⊙ in the source frame) and one sky map per
event under `data/skymaps/<run>/`. The population model is recorded in the
table metadata, so the merger-rate prior cannot drift from the sample. Also
shared by the three runs.

### 3. Schedule

```bash
ugf-scheduler --config config/ultrasat_lcs.ini
ugf-scheduler --config config/ultrasat_allsky.ini
ugf-scheduler --config config/uvex.ini
```

Writes one wrapper script per event under `output/<run>/logs/` and submits it
through the backend named by `[execution] backend`: `condor`, `slurm`,
`parallel` or `dask`. The schedules land in `output/<run>/schedules/<run>/`.

Useful options:

- `--dry-run` writes the wrappers and submits nothing. Read one to check the
  exact m4opt command before launching thousands of jobs.
- `--skip-existing` skips the events that already have a schedule, which is
  what you want when resuming.
- `--backend parallel --n-cores 8` runs a handful of events locally.

A job held for exceeding its memory or disk limit, or failing on the CPLEX
1803 licensing error, is released and retried with a larger request, up to
three times. Jobs that are still held afterwards should be released, not
removed:

```bash
condor_q -hold                       # why they are held
condor_release -constraint 'JobStatus == 5 && Owner == "'"$USER"'"'
```

`condor_rm` throws the work away and the event has to be scheduled from
scratch; `condor_release` puts the job back in the queue with the escalated
request the submit description already carries.

When the queue has drained, check what came back and re-run what did not:

```bash
python scripts/check_runs.py --config config/ultrasat_lcs.ini
python scripts/make_redo.py  --config config/ultrasat_lcs.ini --first 200
ugf-scheduler --config config/ultrasat_lcs.ini \
    --event-table output/ultrasat_lcs/redo.ecsv
```

`check_runs.py` prints the solver statuses per observing run and writes
`output/<run>/crashed.txt`: the events whose plan is missing, unreadable, or
carries the CPLEX 1803 error. `make_redo.py` turns that list back into an
event table the scheduler accepts. These two are operational tooling rather
than steps of the pipeline, which is why they live in `scripts/` and have no
`ugf-` command.

### 4. Post-process

```bash
ugf-postprocess --config config/ultrasat_lcs.ini
ugf-postprocess --config config/ultrasat_allsky.ini
ugf-postprocess --config config/uvex.ini
```

Reads every schedule, computes the kilonova detection probability at the true
source position, and writes one table per run:
`output/<run>/<run>.ecsv`. Copy the three into `results/` to replace the
tables shipped here.

### 5. Numbers, table and figure of the paper

```bash
ugf-paper-results
```

Reads `results/` and writes to `paper_outputs/`:

- `numbers.tex`, one `\newcommand` per number quoted in Sections 4 and 5;
- `mission_comparison.tex`, Table 5;
- `detections_by_area_distance.pdf`, Figure 4.

It prints every number grouped by section, then checks the qualitative
statements of the text against the data; a line marked `CHECK THE TEXT` means
a new run contradicts the prose.

Counts are expected numbers per year at the median merger rate of the table's
population model — 100 Gpc⁻³ yr⁻¹ for FullPop, 68 for PixelPop — read from the
metadata the unpacker wrote. A table that records no model is read as FullPop,
which is what the tables currently in `results/` are.

This step needs no cluster and no CPLEX. Since `results/` holds the three
final tables, a fresh checkout reproduces the whole paper with:

```bash
pip install -e .
ugf-paper-results
```

`notebooks/kilonovae_detection_rate.ipynb` builds the per-run rate tables and
the area–distance figures from the same tables, and
`scripts/figures/` holds the plan and field-of-view figures.

## Tests

```bash
pip install -e ".[test]"
ruff check
ruff format --check
pytest
```

The same checks run as git hooks, plus `codespell` and notebook-output
stripping:

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
```

## Citation

The observing scenarios are archived at
[doi:10.5281/zenodo.22555949](https://doi.org/10.5281/zenodo.22555949)
(PixelPop) and
[doi:10.5281/zenodo.22550048](https://doi.org/10.5281/zenodo.22550048)
(FullPop), and the schedules were produced with
[M4OPT](https://github.com/m4opt/m4opt) 4.0.0.
