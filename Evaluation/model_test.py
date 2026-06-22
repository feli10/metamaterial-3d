import torch
import numpy as np
import os.path as osp
import sys
import matplotlib.pyplot as plt

repo = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, osp.join(repo, "ML_Model"))
sys.path.insert(0, osp.join(repo, "Data_Generation"))

from view_cell import view_cell


def draw_cell(ax, A, threshold=0.0, title=None):
    filled = np.transpose(A > threshold, (2, 1, 0))
    ax.voxels(filled, edgecolor='k', linewidth=0.2)
    ax.set_box_aspect((1, 1, 1))
    if title:
        ax.set_title(title)


device = torch.device("cpu")

model = torch.load(osp.join(repo, "Archive", "e1200-1800", "Freq_FNO.pth"), map_location=device)
model.eval()

d = np.load(osp.join(repo, "dataset.npz"))

for i in range(130, 140):
    x = torch.tensor(d["cells"][i:i+1]).float()        # [1,10,10,10]
    x_np = x.numpy()

    with torch.no_grad():
        x_hat, mean, sph_err_e, mean_dec, sph_err_d = model(x)

    recon = x_hat.view(1, 10, 10, 10).cpu().numpy()
    recon_bin = (recon > 0.5).astype(float)          # x_hat is in (0,1) → threshold

    print("input vf: ", x.mean().item())
    print("recon vf: ", recon_bin.mean())
    print("voxel match:", (recon_bin == x_np).mean())
    print("x_hat range:", recon.min(), recon.max())

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    draw_cell(ax1, x_np[0],  threshold=0.5, title="original")
    draw_cell(ax2, recon[0], threshold=0.5, title="reconstruction")

    plt.tight_layout()
    plt.show()