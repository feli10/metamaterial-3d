# metamaterial-3d

A pipeline for learning the structure–property relationship of 3D metamaterials. Voxelized unit
cells are homogenized with FEM to compute their effective elastic stiffness tensors, and an
FNO-based epistemic-aware autoencoder (EAAE) is trained to predict those properties from geometry
— and, run in reverse, to generate a geometry that achieves prescribed target properties.

On top of that pipeline sits a web tool that lets a user specify target mechanical properties,
generate a lattice unit cell, view it in 3D, and verify the achieved properties by FEM.

**Live tool:** https://metamaterial-sim-hub.poly.edu

---

## Repository Layout

```
ML_Model/           model definition (models_epi_3d.py) + training
                    models_epi_3d_10.py = OLD 10³ class, kept so old checkpoints still load
                    old_checkpoint.py   = load_model(path, old=) handles both vintages
Data_Generation/    dataset generation (generate_cells_15.py) + FEM labeling (run_homog_15.py)
Property_Testing/   homopy_fem_3d.py — homogenize_3d(), the FEM ground truth
Evaluation/         eval scripts; eval_config.py has the OLD=True/False switch
                    poster_eval.py / poster_figures.py produce the result figures
Archive/            model checkpoints — GITIGNORED, see below (d15e200 = current 15³;
                    e1200_100 = old 10³)
web/frontend/       React + TypeScript + Vite SPA
web/backend/        FastAPI service + Dockerfile
```

### Model checkpoints & data

`Archive/` is gitignored — the checkpoints are ~120 MB each and are not in the repository.
Download them from the shared Google Drive and place them so the paths match the `MODELS` registry
in `web/backend/model_service.py` (i.e. `Archive/d15e200/Freq_FNO.pth`,
`Archive/e1200_100/Freq_FNO.pth`):

**Models & datasets:**
https://drive.google.com/drive/folders/1p4i-kJtyHk_xltlKGlrU7r3S4o2pLKem?usp=sharing

---

## Web Tool

### Architecture

```
browser ──HTTPS──> nginx :443 ──┬── /       → static React build
                                └── /api/*  → 127.0.0.1:8000 (Docker container)
```

The backend is a long-lived process, not serverless: `generate()` runs a ~100-iteration Adam
optimization with live autograd, and `evaluate()` runs FEM. The model is loaded once at startup and
kept warm in memory. Frontend and backend share one origin, so no CORS configuration is needed.

Endpoints: `GET /health`, `GET /models`, `POST /generate`, `POST /evaluate`.

### Running locally

First download the checkpoints into `Archive/` (see *Model checkpoints & data* above) — the backend
fails at startup without them.

One-time backend setup (the repo's research code needs torch, which is normally only on the HPC):

```bash
python3 -m venv web/backend/.venv
web/backend/.venv/bin/python -m pip install torch numpy scipy fastapi "uvicorn[standard]"
```

Then, in two terminals:

```bash
# backend  → http://localhost:8000
cd web/backend && .venv/bin/python -m uvicorn main:app --port 8000

# frontend → http://localhost:5173
cd web/frontend && npm install && npm run dev
```

Open http://localhost:5173. Start the backend first so the model dropdown populates.

### Adding a model

Models are registered in one place: the `MODELS` list in `web/backend/model_service.py`. Add a
`ModelSpec` (checkpoint / dataset / latent cache / `baseline` / `grid_dim`, and `old=True` for
pre-15³ checkpoints) and it appears in the dropdown automatically. 

---

## Deploying / Updating the Site

The tool runs on the VM `metamaterial-sim-hub.poly.edu` (RHEL 9, Docker + nginx). SSH requires the
**NYU VPN**.

**Frontend only** (no downtime):

```bash
cd web/frontend && VITE_API_BASE=/api npm run build
rsync -avz --delete dist/ xz5367@metamaterial-sim-hub.poly.edu:~/frontend-dist/
# on the VM:
sudo cp -r ~/frontend-dist/. /usr/share/nginx/html/ && sudo bash -c 'restorecon -R /usr/share/nginx/html'
```

**Backend** (~30s downtime):

```bash
rsync -avzR --exclude='.venv' --exclude='__pycache__' \
  web/backend ML_Model Property_Testing xz5367@metamaterial-sim-hub.poly.edu:~/metamaterial-3d/
# on the VM:
cd ~/metamaterial-3d
sudo docker build -f web/backend/Dockerfile -t metamaterial-api .
sudo docker stop metamaterial-api && sudo docker rm metamaterial-api
sudo docker run -d --name metamaterial-api --restart=unless-stopped -p 127.0.0.1:8000:8000 metamaterial-api
```

Verify after either: `curl -s https://metamaterial-sim-hub.poly.edu/api/health`

---

## Reference

J. Chen, J. Alrumaihi, N. Gupta. "Enabling inverse design of metamaterials via trustworthy and
interpretable representation learning with epistemic uncertainty awareness." *Engineering
Applications of Artificial Intelligence* 177 (2026): 114853.
