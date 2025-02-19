# Astrophysical Merger Rate Calculation

## Overview
`farah.py` is a Python script designed to:
- **Download** the GWTC-3 distribution from the LIGO Document Control Center (DCC).
- **Process** the dataset into input parameters for observing scenarios.
- **Classify** compact binary systems into their subpopulations: **BNS**, **NSBH**, and **BBH**.
- **Compute** the astrophysical merger rates for each subpopulation based on mass distributions.
- **Save** the processed data for further analysis.

## GWTC-3 Distribution
Here a Python script to download the GWTC-3 dataset from LIGO DCC, then process it as the observing scenarios input parameters and calculate the merger rate of each subpopulation.

[farah.py](scenarios/farah.py)

```
python3 scenarios/farah.py --outdir scenarios
```
By default, we use the `scenarios/` folder, but you can specify another directory with `--outdir`.
