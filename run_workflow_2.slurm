#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=bcrv-delta-cpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-199%10  # Only run 10 array tasks concurrently
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10gb
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=leggi014@umn.edu
#SBATCH -e error.ultrasat
#SBATCH -o out.ultrasat

# Load necessary modules or activate environment if needed
# module load python/3.10
# source ~/path/to/venv/bin/activate

# Run ULTRASAT workflow step 2
echo "Running task ID: $SLURM_ARRAY_TASK_ID"

# Pass it to a Python script
python3 run_workflow_2.py --params params-O6.ini --task-id $SLURM_ARRAY_TASK_ID

