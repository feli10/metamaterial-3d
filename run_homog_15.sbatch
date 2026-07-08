#!/bin/bash
#SBATCH --job-name=homog15
#SBATCH --account=torch_pr_183_general
#SBATCH --array=0-0                 # TIMING TEST: just task 0. FULL run: 0-255 (see below)
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                    # 15^3: ~283MB for cells + FEM working memory
#SBATCH --time=00:15:00            # TIMING TEST walltime. FULL run: 01:00:00
#SBATCH --output=/scratch/xz5367/metamaterials/logs/homog15_%A_%a.out

module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/xz5367/metamaterials/penv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# N_TASKS must match the --array size: the script splits 82,944 cells into N_TASKS chunks and
# this task homogenizes chunk SLURM_ARRAY_TASK_ID. With the optimized homopy_fem_3d.py a 15^3
# cell takes ~7 s locally (was >30 min before), so the whole dataset is only ~160 CPU-hours.
#
# TIMING TEST (current settings): --array=0-0 + N_TASKS=4000  =>  task 0 does ~21 cells (~2.5 min).
#   Check the gaps between "cell i done" timestamps in the log to confirm the HPC per-cell time.
#
# FULL RUN: set --array=0-255, --time=01:00:00, and N_TASKS=256 below.
#   => ~324 cells/task x ~7 s ≈ 38 min/task, comfortably under the 1 h walltime.
N_TASKS=4000
python "$HOME/metamaterial-3d/Data_Generation/run_homog_15.py" $N_TASKS
