#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=bcrv-delta-cpu
#SBATCH --time=12:00:00
#SBATCH --array=0-199%10  # Only run 10 array tasks concurrently
#SBATCH --ntasks=1

#SBATCH --mem=10gb
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=leggi014@umn.edu
#SBATCH -e error.ultrasat
#SBATCH -o out.ultrasat
# Load any needed modules or activate virtualenv if necessary
# module load python/3.x
# source ~/venv/bin/activate

echo "Starting ULTRASAT Step 1 on $(hostname)"
echo "Running task ID: $SLURM_ARRAY_TASK_ID"

# Pass it to a Python script
python3 run_workflow_1.py --params params-O6.ini --task-id $SLURM_ARRAY_TASK_ID
echo "Finished ULTRASAT Step 1"

