import numpy as np
import torch
import os.path as osp
import sys

repo = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, osp.join(repo, "ML_Model"))

MODEL_PATH = osp.join(repo, "Archive", "e1200_100", "Freq_FNO.pth")
DATASET_PATH = osp.join(repo, "dataset.npz")
N_PROPS = 22

model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()

d = np.load(DATASET_PATH)
iu = np.triu_indices(6)
y = np.concatenate([d["vfs"][:, None], d["C"][:, iu[0], iu[1]]], axis=1)

# encode all cells, pull the first 22 channels of mid_value
cells = torch.tensor(d["cells"], dtype=torch.float32)
preds = []
with torch.no_grad():
    for i in range(0, len(cells), 64):
        mean, _ = model.Encoder(cells[i:i+64])
        preds.append(mean[:, :N_PROPS, 0, 0, 0].numpy())
pred = np.concatenate(preds, axis=0)  # [N,22]

abs_err = np.abs(pred - y)
rel_err = abs_err / (np.abs(y) + 1e-3)

print(f"{'ch':>3} {'raw|mean':>9} {'pred|mean':>9} {'abs_err':>8} {'rel_err':>8}")
for k in range(N_PROPS):
    print(f"{k:>3} {y[:,k].mean():>9.4f} {pred[:,k].mean():>9.4f} "
          f"{abs_err[:,k].mean():>8.4f} {rel_err[:,k].mean():>8.3f}")

print(f"\noverall mean abs err: {abs_err.mean():.4f}")
print(f"overall mean rel err: {rel_err.mean():.3f}")
print(f"vf (ch0) mean rel err: {rel_err[:,0].mean():.3f}")
print(f"C entries (ch1-21) mean rel err: {rel_err[:,1:].mean():.3f}")