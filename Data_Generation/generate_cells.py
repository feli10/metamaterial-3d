import itertools
import numpy as np
from strut_functions import make_frame, make_strut, combine_struts

DIAGONALS = [
    ((0,0,0),(2,2,2)),
    ((0,0,2),(2,2,0)),
    ((0,2,0),(2,0,2)),
    ((2,0,0),(0,2,2))
]

N = 10
frame = make_frame(N, 1)

cells, labels, vfs = [], [], []
for combo in itertools.product([0, 1, 1.5, 2, 2.5, 3], repeat=4):
    struts = [make_strut(a, b, N, t) for (a, b), t in zip(DIAGONALS, combo)]
    cell = combine_struts([frame] + struts)
    cells.append(cell)
    labels.append(combo)
    vfs.append(cell.sum() / cell.size)

cells  = np.array(cells,  dtype=np.uint8)
labels = np.array(labels, dtype=np.float32)
vfs    = np.array(vfs,    dtype=np.float32)
np.savez_compressed("cells.npz", cells=cells, labels=labels, vfs=vfs)

print(cells.shape, labels.shape)