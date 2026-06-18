#!/bin/bash
#SBATCH --job-name=homog
#SBATCH --account=torch_pr_183_general
#SBATCH --array=0-127
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xz5367/metamaterials/logs/homog_%A_%a.out

module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/xz5367/metamaterials/penv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python "$HOME/metamaterial-3d/Data_Generation/run_homog.py" 128