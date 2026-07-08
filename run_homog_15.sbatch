#!/bin/bash
#SBATCH --job-name=homog15
#SBATCH --account=torch_pr_183_general
#SBATCH --array=0-0                 # TEST: just task 0. FULL run: 0-(N_TASKS-1)
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                    # 15^3: ~283MB for cells + FEM working memory (was 2G at 10^3)
#SBATCH --time=00:30:00            # TEST walltime; raise for the full run
#SBATCH --output=/scratch/xz5367/metamaterials/logs/homog15_%A_%a.out

module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/xz5367/metamaterials/penv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# N_TASKS must match the --array size: the script splits 82,944 cells into N_TASKS chunks and
# this task does chunk SLURM_ARRAY_TASK_ID.
#
# TIMING TEST: keep --array=0-0 and set N_TASKS large so task 0 does only a handful of cells.
#   82,944 / 16,000 ≈ 5 cells for task 0. Read the "cell i done" timestamps in the log to get
#   per-cell time T_hpc, then size the FULL run:  N_TASKS ≈ ceil(82,944 * T_hpc / walltime).
# FULL RUN: set --array=0-(N_TASKS-1) and --time above, and set N_TASKS below to match.
N_TASKS=16000
python "$HOME/metamaterial-3d/Data_Generation/run_homog_15.py" $N_TASKS
