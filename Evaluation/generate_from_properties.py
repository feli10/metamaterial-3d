"""
inverse_design_3d.py  -- pin a target property vector, search the 10 free latent
dims for the highest-confidence geometry, then show it. Runs locally.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os.path as osp
import sys

repo = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, osp.join(repo, "ML_Model"))

CKPT = osp.join(repo, "Archive", "e1200_guard", "Freq_FNO.pth")
DATASET = osp.join(repo, "dataset.npz")
N_PROPS, N_FREE = 22, 10
SPH_ERR_BASELINE = 0.04
THRESHOLD = 0.9995
LR = 5e-2
ITERS = 2000
RESTARTS = 5

model = torch.load(CKPT, map_location="cpu", weights_only=False)
model.device = torch.device("cpu")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

def confidence(mid_value):
    """[1,32,1,1,1] latent -> (fa, geometry). Zero-variance reparameterization
    gives the decoder the deterministic complex latent it trained on."""
    z = model.Encoder.reparameterization(mid_value, torch.zeros_like(mid_value),
        model.modes1, model.modes2, model.modes3)
    x_hat, _, sph_err = model.Decoder(z)
    fa = torch.exp(-sph_err.mean() / SPH_ERR_BASELINE)
    return fa, x_hat

d = np.load(DATASET)
i = 0
iu = np.triu_indices(6)
z_fixed = torch.tensor(np.concatenate([[d["vfs"][i]], d["C"][i][iu]]), dtype=torch.float32)

# gradient ascent on the 10 free dims, a few random restarts, keep the best
best_free, best_fa = None, 0.0
for r in range(RESTARTS):
    z_free = torch.randn(N_FREE, requires_grad=True)
    opt = torch.optim.Adam([z_free], lr=LR)
    for _ in range(ITERS):
        opt.zero_grad()
        mid = torch.cat([z_fixed, z_free]).view(1, 32, 1, 1, 1)
        fa, _ = confidence(mid)
        (-fa).backward()
        opt.step()
        if fa.item() >= THRESHOLD:
            break
    print(f"restart {r}: fa = {fa.item():.5f}")
    if fa.item() > best_fa:
        best_free, best_fa = z_free.detach(), fa.item()

# decode the best latent and view the geometry
with torch.no_grad():
    mid = torch.cat([z_fixed, best_free]).view(1, 32, 1, 1, 1)
    _, x_hat = confidence(mid)
geometry = x_hat.squeeze().numpy()        # 10x10x10 occupancy in [0,1]

ax = plt.figure().add_subplot(projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.voxels(geometry > 0.5, facecolors="#1D9E75", edgecolor="k", linewidth=0.2)
ax.set_title(f"confidence fa = {best_fa:.4f}")
plt.show()