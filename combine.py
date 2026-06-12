import os
import glob
import numpy as np

repo = os.path.dirname(__file__)
d = np.load(os.path.join(repo, "cells.npz"))
cells, labels, vf = d["cells"], d["labels"], d["vf"]

# pre-allocate the full tensor array; we'll fill each row from the partials
C = np.zeros((len(cells), 6, 6), dtype=np.float32)

results = "/scratch/xz5367/metamaterials/results"
for f in glob.glob(os.path.join(results, "part_*.npz")):
    p = np.load(f)
    C[p["idx"]] = p["C"]          # place each task's results in their global rows

out = "/scratch/xz5367/metamaterials/dataset.npz"
np.savez_compressed(out, cells=cells, labels=labels, vf=vf, C=C)

# sanity check: every cell should have a result, so no row of C is all-zero
zero_rows = int((C == 0).all(axis=(1, 2)).sum())
print(f"wrote {out} | C: {C.shape} | unfilled rows: {zero_rows}")