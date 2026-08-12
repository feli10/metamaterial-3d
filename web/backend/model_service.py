"""
model_service.py — the bridge between the research code and the web API.

This is the ONLY file that imports the ML/FEM research code. It:
  1. loads a trained checkpoint once and keeps it warm in memory, and
  2. exposes two plain functions the API calls: generate() and evaluate().

The logic mirrors `Evaluation/interactive_generation.py` (the reference implementation), with two deliberate, CLAUDE.md-sanctioned generalizations for the web tool:
  - generate() accepts ANY subset of the 22 channels (the reference hardcoded 7). The "known" channels are simply the non-null entries of `target`; the rest are auto-filled.
  - generate() returns `filledTarget` (all 22 concrete channels) so the results panel can compare auto-filled channels too.

Multiple models / resolutions: everything model-specific lives in a ModelSpec entry in the MODELS registry below. Grid size is never hardcoded - it is read from the loaded model (`model.im_x`). Adding a new checkpoint (10³, 15³, or anything future) = one registry entry.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# --- make the research code importable ---
_REPO = Path(__file__).resolve().parents[2]  # web/backend/ -> web/ -> repo root
sys.path.insert(0, str(_REPO / "ML_Model"))
sys.path.insert(0, str(_REPO / "Property_Testing"))

from homopy_fem_3d import homogenize_3d, IsotropicMaterial  # noqa: E402
from old_checkpoint import load_model  # noqa: E402  (handles old & new pickles)

# --- optimization hyperparametersy ---
LR = 5e-2
ITERS = 100
THRESHOLD = 0.9995
BASE_MAT = IsotropicMaterial(E=1.0, nu=0.3)

CHANNEL_COUNT = 22
_TRIU = np.triu_indices(6)

@dataclass(frozen=True)
class ModelSpec:
    """
    Everything that differs between one trained model and another.

    Paths are relative to the repo root. `grid_dim` is recorded here so /models can report each model's resolution WITHOUT loading it; it is asserted against model.im_x on load. `old=True` marks a pre-15³ checkpoint pickled against the old model class (load_model then aliases the preserved old class — see ML_Model/old_checkpoint.py).
    """

    id: str          # dropdown id the frontend sends (ModelInfo.id)
    label: str       # what the dropdown shows
    checkpoint: str
    dataset: str
    latent: str
    baseline: float  # SPH_ERR_BASELINE for fa = exp(-(eps_e - baseline))
    grid_dim: int
    old: bool = False

# The model registry
MODELS: list[ModelSpec] = [
    ModelSpec(
        id="s15v1",
        label="s15v1 (15³)",
        checkpoint="Archive/d15e200/Freq_FNO.pth",
        dataset="dataset_15.npz",
        latent="Evaluation/latent_data/Freq_FNO_15_training_latent_data.txt",
        baseline=0.13,
        grid_dim=15,
        old=False,
    ),
    # The old 10³ model (superseded by s15v1, kept as a second selectable option). `old=True` loads it via the models_epi_3d_10 aliasing (it was pickled against the old model class).
    ModelSpec(
        id="s10v1",
        label="s10v1 (10³)",
        checkpoint="Archive/e1200_100/Freq_FNO.pth",
        dataset="dataset.npz",
        latent="Evaluation/latent_data/Freq_FNO_training_latent_data.txt",
        baseline=0.04,
        grid_dim=10,
        old=True,
    ),
]

_MODELS_BY_ID = {m.id: m for m in MODELS}

class ModelService:
    """
    One loaded model plus its dataset/latent caches. Holds the model warm in memory.

    generate() runs a ~100-iteration Adam optimization on a single latent; evaluate() runs FEM homogenization. Both are single-sample (batch 1), so they broadcast through the model correctly (see eval-workflow memory: multi-sample would need the training batch size). A lock serializes calls since there is one shared model object.
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._lock = threading.Lock()

        # Load the model once, freeze it for inference.
        model = load_model(str(_REPO / spec.checkpoint), old=spec.old, map_location="cpu")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        self.N = int(model.im_x)
        assert self.N == spec.grid_dim, (
            f"{spec.id}: model.im_x={self.N} != spec.grid_dim={spec.grid_dim}"
        )

        # Training property vectors (22-ch)
        d = np.load(_REPO / spec.dataset)
        self.all_props = np.column_stack(
            [d["vfs"], d["C"][:, _TRIU[0], _TRIU[1]]]
        ).astype(np.float32)  # (num_cells, 22)

        # Training latents
        latent = np.loadtxt(_REPO / spec.latent, delimiter=",").astype(np.float32)
        self.train_props = latent[:, :CHANNEL_COUNT]
        self.train_free = latent[:, CHANNEL_COUNT:]

    # -- internals ---

    def _decode(self, mid: torch.Tensor):
        """Latent (1,32,1,1,1) -> (eps_e, fa, x_hat). Mirrors interactive_generation.evaluate()."""
        m = self.model
        z = m.Encoder.reparameterization(
            mid, torch.zeros_like(mid), m.modes1, m.modes2, m.modes3
        )
        x_hat, _, sph_err = m.Decoder(z, self.N, self.N, self.N)
        eps_e = sph_err.reshape(-1).mean()
        fa = torch.exp(-(eps_e - self.spec.baseline))
        return eps_e, fa, x_hat

    # -- public API ---
    def generate(self, target: list[float | None], on_progress=None) -> dict:
        """target: 22 entries, each a number or None (=auto). Returns voxels/filledTarget/fa/epsE."""
        if len(target) != CHANNEL_COUNT:
            raise ValueError(f"target must have {CHANNEL_COUNT} channels, got {len(target)}")
        exposed = [i for i, v in enumerate(target) if v is not None]
        if not exposed:
            raise ValueError("at least one channel must be set (all-auto is not a valid target)")
        non_exposed = [i for i in range(CHANNEL_COUNT) if i not in exposed]

        # Build the concrete target. Start with the user's known channels...
        t = np.zeros(CHANNEL_COUNT, dtype=np.float32)
        for i in exposed:
            t[i] = float(target[i])

        # ...then fill the unknown channels from the training cell closest on the KNOWN channels (generalized from the reference's fixed exposed set).
        exp = np.array(exposed)
        d_props = np.sum(
            (self.all_props[:, exp] - t[exp]) ** 2 / (t[exp] + 1e-3) ** 2, axis=1
        )
        nearest_cell = int(np.argmin(d_props))
        for i in non_exposed:
            t[i] = self.all_props[nearest_cell, i]

        # Seed the 10 free latent dims from the nearest training latent.
        dist = np.sum((self.train_props - t) ** 2 / (t + 1e-3) ** 2, axis=1)
        nearest = int(np.argmin(dist))

        with self._lock:
            z_fixed = torch.tensor(t, dtype=torch.float32)
            z_free = torch.tensor(self.train_free[nearest], requires_grad=True)

            opt = torch.optim.Adam([z_free], lr=LR)
            fa_val, eps_val, x_hat = 0.0, 0.0, None
            for i in range(ITERS):
                opt.zero_grad()
                mid = torch.cat([z_fixed, z_free]).view(1, 32, 1, 1, 1)
                eps_e, fa, x_hat = self._decode(mid)
                eps_e.backward()
                opt.step()
                fa_val, eps_val = fa.item(), eps_e.item()
                if on_progress is not None:
                    on_progress(i + 1, ITERS, fa_val)
                if fa_val >= THRESHOLD:
                    break

            cell = (x_hat.squeeze().detach().numpy() > 0.5).astype(np.float64)

        return {
            # Flatten X-fastest to match types.ts
            "voxels": cell.flatten(order="F").astype(int).tolist(),
            "filledTarget": t.astype(float).tolist(),
            "fa": fa_val,
            "epsE": eps_val,
        }

def evaluate(voxels: list[int], on_progress=None) -> dict:
    """
    voxels: N^3 flat 0/1 (X-fastest). Returns achieved 22-ch homogenized properties.

    Model-INDEPENDENT: FEM homogenization uses only the geometry + base material, so the grid size N is inferred from len(voxels) (matches types.ts EvaluateRequest = {voxels}).
    """
    n = round(len(voxels) ** (1 / 3))
    if n ** 3 != len(voxels):
        raise ValueError(f"{len(voxels)} voxels is not a perfect cube (N³)")
    # Invert the same X-fastest flattening generate() used, so homogenize sees the identical array the model produced.
    cell = np.asarray(voxels, dtype=np.float64).reshape((n, n, n), order="F")
    C = homogenize_3d(n, n, n, BASE_MAT, density_field=cell)
    achieved = np.concatenate([[cell.mean()], C[_TRIU[0], _TRIU[1]]]).astype(float)
    return {"achieved": achieved.tolist()}

# --- lazy, cached model manager ---
_services: dict[str, ModelService] = {}
_services_lock = threading.Lock()

def list_models() -> list[dict]:
    """For GET /models. Reports id/label/gridDim without loading the checkpoints."""
    return [{"id": m.id, "label": m.label, "gridDim": m.grid_dim} for m in MODELS]

def get_service(model_id: str) -> ModelService:
    """Return the (cached) ModelService for a model id, loading it on first use."""
    if model_id not in _MODELS_BY_ID:
        raise KeyError(f"unknown model id: {model_id!r}")
    with _services_lock:
        if model_id not in _services:
            _services[model_id] = ModelService(_MODELS_BY_ID[model_id])
        return _services[model_id]
