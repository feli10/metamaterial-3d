import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scipy.spatial import distance
import sys

repo = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, osp.join(repo, "ML_Model"))

from utils_epi_3d import CombinedDataset

# == CONFIG ==========

device = torch.device("cpu")
batch_size = 16
SPH_ERR_BASELINE = 0.04 
N_AUG = 3
 
model_file = osp.join(repo, "Archive", "e1200", "Freq_FNO.pth")
latent_data_dir = osp.join(repo, "Evaluation", "latent_data")
figure_dir = osp.join(repo, "Evaluation", "figures")
train_latent_cache = osp.join(latent_data_dir, "Freq_FNO_training_latent_data.txt")

# == DATA ==========
 
d = np.load(osp.join(repo, "dataset.npz"))
X = d["cells"].astype(np.float32)                        
iu = np.triu_indices(6)
C_flat = d["C"][:, iu[0], iu[1]]
y = np.concatenate([d["vfs"][:, None], C_flat], axis=1).astype(np.float32)

dataset = CombinedDataset(X, y)
train_dataset, test_dataset = train_test_split(
    dataset, test_size=0.1, random_state=42)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# == MODEL SETUP ==========

model = torch.load(model_file, map_location=device, weights_only=False)
model.device = device
model.to(device)
model.eval()

def to_latent(mean):
    """[B, latent, 1, 1, 1] -> [B, latent] numpy."""
    return mean.reshape(mean.shape[0], -1).numpy()
 
def confidence(sph_err):
    """[B, ...] decoder spherical error -> [B] confidence score."""
    per_sample = sph_err.reshape(sph_err.shape[0], -1).mean(axis=1)
    return np.exp(-(per_sample - SPH_ERR_BASELINE))

# == TRAIN LATENTS ========

use_cache = (
    osp.exists(train_latent_cache)
    and osp.getmtime(train_latent_cache) > osp.getmtime(model_file)
)
if use_cache:
    training_latent_data = np.loadtxt(train_latent_cache, delimiter=',').astype(np.float32)
else:
    training_latent_data = []
    for x, _ in tqdm(train_loader, desc="train latents"):
        x = x.float().to(device)
        with torch.no_grad():
            _, mean, _, _, _ = model(x)
        training_latent_data.append(to_latent(mean))
    training_latent_data = np.vstack(training_latent_data)
    np.savetxt(train_latent_cache, training_latent_data, delimiter=',', fmt='%.6f')
print("training_latent_data:", training_latent_data.shape)

# == TEST LATENTS ========

test_latent_data = []
test_sph_err = []
 
for x, _ in tqdm(test_loader, desc="test latents"):
    x = x.float().to(device)
    with torch.no_grad():
        _, mean, sph_err_e, _, sph_err_d = model(x)
 
    # the real test point
    test_latent_data.append(to_latent(mean))
    test_sph_err.append(confidence(sph_err_d.cpu().numpy()))
 
    # push it progressively off-manifold by perturbing one latent dim,
    # re-decode, and record the *new* decoder spherical error.
    min_noise = 0.0
    for _ in range(N_AUG):
        noise = torch.zeros_like(mean)
        b = mean.shape[0]
        rows = torch.arange(b)
        cols = torch.randint(0, mean.shape[1], (b,))
        noise[rows, cols, 0, 0, 0] = 0.1 * torch.rand(b) + min_noise
        mean2 = mean + noise
        with torch.no_grad():
            z = model.Encoder.reparameterization(
                mean2, sph_err_e, model.modes1, model.modes2, model.modes3)
            _, _, sph_err_d2 = model.Decoder(z)
        test_latent_data.append(to_latent(mean2))
        test_sph_err.append(confidence(sph_err_d2.cpu().numpy()))
        min_noise += 0.1
 
test_latent_data = np.concatenate(test_latent_data, axis=0)
confidence_score = np.concatenate(test_sph_err, axis=0)
print("test_latent_data:", test_latent_data.shape,
      "confidence_score:", confidence_score.shape)

# == PLOT ========

dists = distance.cdist(test_latent_data, training_latent_data, 'euclidean')
min_dist = dists.min(axis=1)
print("min_dist:", min_dist.shape)
 
plt.figure()
plt.scatter(min_dist, confidence_score, alpha=0.5)
plt.xlabel('Distance to Training Data')
plt.ylabel('Confidence score')
plt.grid(True)
plt.savefig(osp.join(figure_dir, "Freq_FNO_confidence_vs_min_dist.png"))
np.savetxt(osp.join(latent_data_dir, "Freq_FNO_confidence_vs_min_dist.txt"),
           np.column_stack((min_dist, confidence_score)), delimiter=',', fmt='%.6f')
plt.show()