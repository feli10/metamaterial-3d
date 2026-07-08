import itertools
import numpy as np
from strut_functions import make_frame, make_strut, combine_struts

DIAGONALS = [
    ((0,0,0),(2,2,2)),
    ((0,0,2),(2,2,0)),
    ((0,2,0),(2,0,2)),
    ((2,0,0),(0,2,2))
]

STRAIGHTS = [
    ((0,0,0), (2,0,0)),
    ((0,0,2), (2,0,2)),
    ((0,2,0), (2,2,0)),
    ((0,2,2), (2,2,2)),

    ((0,0,0), (0,2,0)),
    ((0,0,2), (0,2,2)),
    ((2,0,0), (2,2,0)),
    ((2,0,2), (2,2,2)),

    ((0,0,0), (0,0,2)),
    ((0,2,0), (0,2,2)),
    ((2,0,0), (2,0,2)),
    ((2,2,0), (2,2,2))
]

N = 15
s_thick = [1, 2, 3, 4]
d_thick = [0, 1.5, 2, 2.5, 3, 3.5]

precomputed_straights = {}
for i, (a, b) in enumerate(STRAIGHTS):
    for t in s_thick:
        precomputed_straights[(i, t)] = make_strut(a, b, N, t)

precomputed_diags = {}
for i, (a, b) in enumerate(DIAGONALS):
    for t in d_thick:
        precomputed_diags[(i, t)] = make_strut(a, b, N, t)

cells, labels, vfs = [], [], []
for s_combo in itertools.product(s_thick, repeat=3):
    for d_combo in itertools.product(d_thick, repeat=4):

        diag_struts = [precomputed_diags[(i, t)] for i, t in enumerate(d_combo)]
        
        straight_struts = []
        for group_idx, t in enumerate(s_combo):
            for str_idx in range(group_idx * 4, (group_idx + 1) * 4):
                straight_struts.append(precomputed_straights[(str_idx, t)])
    
        # cast to uint8 immediately: combine_struts returns int64 (~27 KB/cell), so keeping
        # 82,944 of them in a list would peak ~2.2 GB. uint8 (0/1 values) drops that to ~280 MB.
        cell = combine_struts(diag_struts + straight_struts).astype(np.uint8)

        cells.append(cell)
        labels.append(s_combo + d_combo)
        vfs.append(cell.sum() / cell.size)

cells  = np.array(cells,  dtype=np.uint8)
labels = np.array(labels, dtype=np.float32)
vfs    = np.array(vfs,    dtype=np.float32)
np.savez_compressed("cells_15.npz", cells=cells, labels=labels, vfs=vfs)

print(cells.shape, labels.shape)