#!/bin/bash
#SBATCH --job-name=train_epi
#SBATCH --account=torch_pr_183_general
#SBATCH --gres=gpu:1                # request 1 GPU (Torch has H200/L40S/H100/A100/RTX Pro 6000)
#SBATCH --cpus-per-task=4           # a few CPU cores for the data loader
#SBATCH --mem=32G                   # dataset_15 cells are ~1.1 GB float32 + overhead
#SBATCH --time=12:00:00             # starting estimate — measure per-epoch time first, then tune
#SBATCH --output=/scratch/xz5367/metamaterials/logs/train_%j.out
# Optional: pin a GPU type, e.g.  #SBATCH --constraint=a100   (omit to take any available GPU)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DATASET_PATH=/scratch/xz5367/metamaterials/dataset_15.npz
export OUTPUT_DIR=/scratch/xz5367/metamaterials
export LD_LIBRARY_PATH=/scratch/xz5367/metamaterials/penv/lib:$LD_LIBRARY_PATH

/scratch/xz5367/metamaterials/penv/bin/python "$HOME/metamaterial-3d/ML_Model/train_epi_3d.py" "$@"
