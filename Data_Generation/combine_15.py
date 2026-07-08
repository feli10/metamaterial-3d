import os
import glob
import numpy as np

# 15^3 dataset version of combine.py: stitches the per-task part_*.npz homogenization results
# (from run_homog_15.py) back into one labeled dataset. Differences from the 10^3 version are
# just the input cells file, the results dir, and the output filename.

repo = os.path.dirname(os.path.dirname(__file__))
d = np.load(os.path.join(repo, "cells_15.npz"))
cells, labels, vfs = d["cells"], d["labels"], d["vfs"]

# pre-allocate the full tensor array; we'll fill each row from the partials
C = np.zeros((len(cells), 6, 6), dtype=np.float32)

results = "/scratch/xz5367/metamaterials/results_15"
parts = glob.glob(os.path.join(results, "part_*.npz"))
for f in parts:
    p = np.load(f)
    C[p["idx"]] = p["C"]  # place each task's results in their global rows

out = "/scratch/xz5367/metamaterials/dataset_15.npz"
np.savez_compressed(out, cells=cells, labels=labels, vfs=vfs, C=C)

# sanity check: every cell should have a result, so no row of C is all-zero
zero_rows = int((C == 0).all(axis=(1, 2)).sum())
print(f"wrote {out} | parts merged: {len(parts)} | C: {C.shape} | unfilled rows: {zero_rows}")
