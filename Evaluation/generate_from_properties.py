"""
generate_from_properties.py -- pin a target property vector, search the 10 free
latent dims for the lowest epistemic error (highest confidence), then show it.
Confidence matches Evaluation/confidence_test.py exactly. Runs locally.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os.path as osp
import sys

repo = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, osp.join(repo, "ML_Model"))

MODEL_PATH = osp.join(repo, "Archive", "e1200", "Freq_FNO.pth")
DATASET_PATH = osp.join(repo, "dataset.npz")
LATENT_PATH = osp.join(repo, "Evaluation", "latent_data", "Freq_FNO_training_latent_data.txt")
SPH_ERR_BASELINE = 0.04 
THRESHOLD = 0.9995
LR = 5e-2
ITERS = 100

model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.device = torch.device("cpu")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

def evaluate(mid_value):
    z = model.Encoder.reparameterization(mid_value, torch.zeros_like(mid_value),
        model.modes1, model.modes2, model.modes3)
    x_hat, _, sph_err = model.Decoder(z)
    eps_e = sph_err.reshape(-1).mean()
    fa = torch.exp(-(eps_e - SPH_ERR_BASELINE))
    return eps_e, fa, x_hat

d = np.load(DATASET_PATH)
cell_idx = 2

# with torch.no_grad():
#     mean, _ = model.Encoder(torch.tensor(d["cells"][2:3]).float())
# test = [round(float(mean[0][k][0][0][0]),4) for k in range(22)]
# print("encoder ch[:22]:", test)

# z_fixed = torch.tensor(test)

z_fixed = torch.tensor(np.concatenate([[d["vfs"][cell_idx]], 
    d["C"][cell_idx][np.triu_indices(6)]]), dtype=torch.float32)

with torch.no_grad():
    mean, _ = model.Encoder(torch.tensor(d["cells"][2:3]).float())
print("encoder ch[:22]:", [round(float(mean[0][k][0][0][0]),4) for k in range(22)])
print("raw properties: ", z_fixed.numpy())

# find the closest training latent vector and set the free channels to it
training_latent_data = np.loadtxt(LATENT_PATH, delimiter=",").astype(np.float32)
train_props = training_latent_data[:, :22]
target = z_fixed.numpy()
dist = np.sum((train_props - target)**2 / (target + 1e-3)**2, axis=1)
nearest = np.argsort(dist)[0]
z_free = torch.tensor(training_latent_data[nearest, 22:], requires_grad=True)

opt = torch.optim.Adam([z_free], lr=LR)

for _ in range(ITERS):
    opt.zero_grad()
    mid = torch.cat([z_fixed, z_free]).view(1, 32, 1, 1, 1)
    eps_e, fa, x_hat = evaluate(mid)
    eps_e.backward() 
    opt.step()
    if fa.item() >= THRESHOLD:
        break

print(f"fa = {fa.item():.5f}  (eps_e = {eps_e.item():.4f})")
geometry = x_hat.squeeze().detach().numpy()

ax = plt.figure().add_subplot(projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.voxels(geometry > 0.5, facecolors="#1D9E75", edgecolor="k", linewidth=0.2)
plt.show()