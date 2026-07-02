import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os.path as osp
import sys

repo = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, osp.join(repo, "ML_Model"))
sys.path.insert(0, osp.join(repo, "Property_Testing"))

from homopy_fem_3d import homogenize_3d, IsotropicMaterial

MODEL_PATH = osp.join(repo, "Archive", "e1200_100", "Freq_FNO.pth")
DATASET_PATH = osp.join(repo, "dataset.npz")
LATENT_PATH = osp.join(repo, "Evaluation", "latent_data", "Freq_FNO_training_latent_data.txt")

SPH_ERR_BASELINE = 0.04
THRESHOLD = 0.9995
LR = 5e-2
ITERS = 100
cell_idx = 100

BASE_MAT = IsotropicMaterial(E=1.0, nu=0.3)

# Exposed property channels: vf + the six diagonal stiffnesses
EXPOSED = [0, 1, 7, 12, 16, 19, 21]

prop_desc = {
    0:  "vf · volume fraction",
    1:  "C11 · axial stiffness x",
    7:  "C22 · axial stiffness y",
    12: "C33 · axial stiffness z",
    16: "C44 · shear stiffness yz",
    19: "C55 · shear stiffness xz",
    21: "C66 · shear stiffness xy",
}

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
triu = np.triu_indices(6)
prop_labels = ["vf"] + [f"C{i+1}{j+1}" for i, j in zip(*triu)]
all_props = np.column_stack([d["vfs"], d["C"][:, triu[0], triu[1]]]).astype(np.float32)  # (N, 22)
prop_min, prop_max = all_props.min(0), all_props.max(0)
ref = all_props[cell_idx].copy()

training_latent_data = np.loadtxt(LATENT_PATH, delimiter=",").astype(np.float32)
train_props = training_latent_data[:, :22]

exposed = np.array(EXPOSED)
non_exposed = [c for c in range(22) if c not in EXPOSED]


def generate(target, progress=None):
    target = target.copy()

    # Fill non-exposed property channels from the training cell closest on the exposed channels
    d_props = np.sum((all_props[:, exposed] - target[exposed])**2
                     / (target[exposed] + 1e-3)**2, axis=1)
    target[non_exposed] = all_props[np.argmin(d_props), non_exposed]

    # Seed the 10 free latent dims from the nearest training latent (rough init only).
    dist = np.sum((train_props - target)**2 / (target + 1e-3)**2, axis=1)
    nearest = np.argmin(dist)

    z_fixed = torch.tensor(target, dtype=torch.float32)
    z_free = torch.tensor(training_latent_data[nearest, 22:], requires_grad=True)

    # Optimize the free dims to maximize decoder familiarity (minimize eps_e).
    opt = torch.optim.Adam([z_free], lr=LR)
    for i in range(ITERS):
        opt.zero_grad()
        mid = torch.cat([z_fixed, z_free]).view(1, 32, 1, 1, 1)
        eps_e, fa, x_hat = evaluate(mid)
        eps_e.backward()
        opt.step()
        if progress is not None:
            progress(i, fa.item())
        if fa.item() >= THRESHOLD:
            break

    cell = (x_hat.squeeze().detach().numpy() > 0.5).astype(np.float64)
    return cell, fa.item(), eps_e.item()

# --- widget UI ---
fig = plt.figure(figsize=(10, 6))
ax3d = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(bottom=0.42)
warn_txt = fig.text(0.05, 0.97, "", color="crimson", fontsize=9, va="top")
status_txt = fig.text(0.7, 0.85, "", color="gray", fontsize=10)

last_cell = [None]

# One slider per exposed channel. Limits are padded beyond the training range so the user can
# drag out-of-distribution to trigger the warning. Guard against a degenerate range (a channel
# that's constant across the dataset, which would make valmin == valmax).
sliders = {}
for k, ch in enumerate(EXPOSED):
    sax = fig.add_axes([0.25, 0.05 + 0.045 * k, 0.6, 0.03])
    span = prop_max[ch] - prop_min[ch]
    pad = 0.5 * span if span > 0 else max(abs(ref[ch]), 0.01)
    sliders[ch] = Slider(sax, prop_desc.get(ch, prop_labels[ch]),
                         prop_min[ch] - pad, prop_max[ch] + pad, valinit=ref[ch])
    sliders[ch].label.set_fontsize(8)   # descriptive labels are long; shrink to fit


def check_range(_=None):
    msgs = [f"{prop_labels[ch]}={s.val:.3g} outside training "
            f"[{prop_min[ch]:.3g}, {prop_max[ch]:.3g}]"
            for ch, s in sliders.items()
            if s.val < prop_min[ch] or s.val > prop_max[ch]]
    warn_txt.set_text("\n".join(msgs))
    fig.canvas.draw_idle()


for s in sliders.values():
    s.on_changed(check_range)


def on_generate(event):
    target = ref.copy()
    for ch, s in sliders.items():
        target[ch] = s.val
    check_range()

    def progress(i, fa):
        if i % 5 == 0:                       # throttle: repaint every 5 iters
            status_txt.set_text(f"Generating… {i+1}/{ITERS}  fa={fa:.4f}")
            fig.canvas.draw()
            fig.canvas.flush_events()

    status_txt.set_text("Generating…")
    fig.canvas.draw()
    fig.canvas.flush_events()

    cell, fa, eps_e = generate(target, progress)
    last_cell[0] = cell
    status_txt.set_text("")
    geometry = np.transpose(cell, (2, 1, 0))   # plotting convention only
    ax3d.clear()
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.voxels(geometry > 0.5, facecolors="#1D9E75", edgecolor="k", linewidth=0.2)
    ax3d.set_title(f"fa = {fa:.5f}   eps_e = {eps_e:.4f}")
    fig.canvas.draw_idle()


def on_evaluate(event):
    cell = last_cell[0]
    if cell is None:
        return
    status_txt.set_text("Homogenizing…")
    fig.canvas.draw()
    fig.canvas.flush_events()
    # Homogenize the generated geometry to get its *physical* properties (ground truth), which
    # may not exactly equal the target you dialed in. Same call as the dataset's run_homog.py.
    C = homogenize_3d(10, 10, 10, BASE_MAT, density_field=cell)
    achieved = np.concatenate([[cell.mean()], C[triu[0], triu[1]]]).astype(np.float32)
    status_txt.set_text("")
    print(f"{'chan':>5} {'target':>9} {'achieved':>9} {'err':>8}")
    for ch, s in sliders.items():
        print(f"{prop_labels[ch]:>5} {s.val:9.4f} {achieved[ch]:9.4f} {achieved[ch]-s.val:8.4f}")


btn_gen = Button(fig.add_axes([0.7, 0.9, 0.2, 0.05]), "Generate")
btn_gen.on_clicked(on_generate)

btn_eval = Button(fig.add_axes([0.7, 0.75, 0.2, 0.05]), "Evaluate")
btn_eval.on_clicked(on_evaluate)

on_generate(None)   # initial render; remove to start blank
plt.show()