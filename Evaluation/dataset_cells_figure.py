import os.path as osp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_repo = osp.dirname(osp.dirname(osp.abspath(__file__)))
d = np.load(osp.join(_repo, "dataset_15.npz"))
cells, vfs, labels = d["cells"], d["vfs"], d["labels"]

order = np.argsort(vfs)
picks = [order[int(0.12 * len(order))],   # sparse
         order[int(0.45 * len(order))],   # medium
         order[int(0.80 * len(order))]]   # dense

FACE = "#3E6DA6"
EDGE = "#0d193a"

fig = plt.figure(figsize=(12, 4.2))
for j, idx in enumerate(picks):
    cell = cells[idx].astype(bool)
    ax = fig.add_subplot(1, 3, j + 1, projection="3d")
    ax.voxels(cell, facecolors=FACE, edgecolor=EDGE, linewidth=0.15)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    ax.set_title(f"vf = {vfs[idx]:.2f}", fontsize=15, color="#1B2430", y=0.02)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0)
out = osp.join(_repo, "Evaluation", "dataset_cells_strip.png")
fig.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
print("wrote", out, "| picks vf:", [round(float(vfs[i]), 3) for i in picks])
