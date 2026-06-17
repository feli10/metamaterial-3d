#!/bin/bash
#SBATCH --job-name=train_epi
#SBATCH --account=torch_pr_183_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xz5367/metamaterials/logs/train_%j.out

module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/xz5367/metamaterials/penv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export DATASET_PATH=/scratch/xz5367/metamaterials/dataset.npz
export OUTPUT_DIR=/scratch/xz5367/metamaterials

python "$HOME/metamaterial-3d/ML_Model/train_epi_3d.py"