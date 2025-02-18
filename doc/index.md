# M4OPT Installation Guide

## Introduction
M4OPT is an open-source toolkit for multi-facility scheduling of astrophysics observing campaigns. It focuses on extremely rapid follow-up of gravitational wave (GW) and neutrino events with heterogeneous networks of space and ground-based observatories.

This installation guide covers setting up M4OPT, dependencies, and the UVEX mission.

---

## 1. Setting Up M4OPT

### Create and Activate a Conda Environment
```bash
conda create --name m4opt_env python=3.11
conda activate m4opt_env
```

### Clone and Install M4OPT
```bash
git clone https://github.com/m4opt/m4opt.git
cd m4opt/
pip install -e .
cd ..
```

### Verify Installation
```bash
m4opt --help
m4opt schedule --help
m4opt schedule --mission ultrasat
```

---
## 2. Install CPLEX

[Install CPLEX](https://m4opt.readthedocs.io/en/latest/install/cplex.html)

## 3. Installing Additional Dependencies

```bash
pip install pandas
pip install scipy==1.13.1
```

**Note:** `interp2d` has been removed in SciPy 1.14.0. For legacy support, consider using:
- `RectBivariateSpline` for regular grids
- `bisplrep`/`bisplev` for scattered 2D data

For more details, visit [SciPy Interpolation Guide](https://scipy.github.io/devdocs/tutorial/interpolate/interp_transition_guide.html).

---

## 4. Verifying Installation
```bash
m4opt schedule --help
```

For additional documentation, visit [M4OPT ReadTheDocs](https://m4opt.readthedocs.io/en/latest/install/index.html).
