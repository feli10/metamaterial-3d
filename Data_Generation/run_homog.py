import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "Property_Testing"))
from homopy_fem_3d import homogenize_3d, IsotropicMaterial

# --- figure out which slice of cells THIS array task is responsible for ---
# Slurm sets SLURM_ARRAY_TASK_ID (0, 1, 2, ...) for each task in the array.
# We pass the total number of tasks as a command-line argument so the script
# knows how many pieces to split the data into.
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
n_tasks = int(sys.argv[1])

# --- load all the cells (the whole file; cheap, it's small) ---
repo = os.path.dirname(os.path.dirname(__file__))
cells = np.load(os.path.join(repo, "cells.npz"))["cells"]

# Split the indices [0, 1, ..., 1295] into n_tasks balanced chunks,
# then pick out the chunk belonging to this task.
idx = np.array_split(np.arange(len(cells)), n_tasks)[task_id]

# --- homogenize each cell in this task's slice ---
mat = IsotropicMaterial(E=1.0, nu=0.3)
C = np.zeros((len(idx), 6, 6), dtype=np.float32)
for j, i in enumerate(idx):
    C[j] = homogenize_3d(10, 10, 10, mat, density_field=cells[i].astype(float))
    print(f"task {task_id}: cell {i} done", flush=True)

# --- save this task's partial result to scratch ---
# We store the global indices alongside the tensors so the combine step
# can place each result in the right row later.
outdir = "/scratch/xz5367/metamaterials/results"
os.makedirs(outdir, exist_ok=True)
np.savez(os.path.join(outdir, f"part_{task_id:04d}.npz"), idx=idx, C=C)
print(f"task {task_id}: wrote {len(idx)} results", flush=True)