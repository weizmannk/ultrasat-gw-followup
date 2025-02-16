# ULTRASAT Follow-up Pipeline for Electromagnetic Counterparts of Gravitational Waves


## Overview
This pipeline automates the scheduling and follow-up of **gravitational wave (GW) events** using the **ULTRASAT** mission. It processes GW localization maps, filters viable events, determines optimal exposure times, and submits observation jobs via **HTCondor**.

---

## Installation

### Installing M4OPT
M4OPT is use for **mission optimization and scheduling**.  
Follow the installation instructions:

[M4OPT Documentation](doc/index.md)


---

## ULTRASAT Workflow Execution

The pipeline consists of three main workflow execution scripts, each responsible for a specific stage of the ULTRASAT GW follow-up process.

### **1. Initial Processing and Filtering**
This script (`run_workflow_1.py`) automates the first stage of the ULTRASAT workflow, including:
- **Parsing configuration parameters** from a `.ini` file.
- **Filtering simulated GW events** (`localization_cut_and_batch.py`):
  - Filters events based on **sky localization area (`max_area`)**.
  - Retains events **below the threshold** and groups them into batches.
  - Uses `classify_populations` to categorize events into **BNS, NSBH, and BBH**.
- **Processing GW localization maps** (`max-texp-by-sky-loc.py`):
  - Reads **GW localization maps** and extracts sky coverage probabilities.
  - Computes the **maximum exposure time (`texp_max`)** within the **90% credible region**.
  - Filters out events exceeding a **maximum allowed exposure time (`max_texp`)**.
  - Outputs a **list of viable events** for follow-up.
- **Submitting batch jobs to HTCondor**:
  - Each batch file is processed independently.
  - Jobs are submitted using an automated HTCondor submission script.

#### **Command:**
```
python3 run_workflow_1.py --params /path/to/params_file.ini
```

### **2. Observation Scheduling and Job Submission**
This script (`run_workflow_2.py`) automates the second stage of the ULTRASAT workflow, focusing on scheduling observations for selected gravitational wave events.

- **Reading observation parameters** from a `.ini` configuration file.
- **Preprocessing exposure times** using `texp_cut_and_batch.py`:
  - Extracts exposure time constraints from the configuration file.
  - Filters out events that exceed the mission-defined exposure limits.
  - Generates batch files with viable observation targets.
- **Generating ULTRASAT observation schedules with `M4OPT`**:
  - Iterates over event batch files.
  - Assigns observation time slots based on ULTRASAT’s operational constraints.
- **Submitting jobs to HTCondor**:
  - Each batch file is processed independently.
  - Jobs are submitted using an automated HTCondor submission script.
- **Logging execution details** for monitoring and debugging:
  - Outputs detailed logs to track scheduling progress.
  - Stores logs in a specified directory for post-processing analysis.

#### **Command:**
```
python3 run_workflow_2.py --params /path/to/params_file.ini
```

### **3. Follow-up Coverage and Visualization**
This script (`run_workflow_3.py`) automates the third stage of the ULTRASAT workflow, focusing on computing follow-up coverage for gravitational wave events and generating visualizations.

- **Reading observation parameters** from a `.ini` configuration file.
- **Computing ULTRASAT follow-up coverage** using `compute_tiling.py`:
  - Processes selected gravitational wave events.
  - Determines ULTRASAT’s coverage for each event.
  - Generates observation plans based on mission constraints.
- **Generating statistical plots and visualizations** using `make-coverage-plots.py`:
  - Creates sky maps illustrating ULTRASAT’s coverage.
  - Produces histograms and summary statistics for observation efficiency.
  - Outputs visual reports for scientific analysis.
- **Logging execution details** for monitoring and debugging:
  - Saves coverage reports for post-processing.
  - Stores logs to track execution status.

#### **Command:**
```
python3 run_workflow_3.py --params /path/to/params_file.ini 
```