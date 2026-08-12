# Backend image for the metamaterial inverse-design API.

# IMPORTANT: build from the REPO ROOT (not web/backend/), because the image bundles the research code + model assets that live above this directory:

#   docker build -f web/backend/Dockerfile -t metamaterial-api .
#   docker run --rm -p 8000:8000 metamaterial-api

# The COPY paths below are relative to the repo root and reproduce the repo layout under /app, so model_service.py's `_REPO = parents[2]` resolves to /app and finds every asset.

FROM python:3.12-slim

# CPU-only torch first (its own layer, cached across rebuilds). The CPU index avoids the multi-GB CUDA wheel — we do inference on CPU (32-dim latent, 15^3 mesh; no GPU needed).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# The rest of the Python deps
COPY web/backend/requirements.txt /app/web/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/web/backend/requirements.txt

WORKDIR /app

# Research code
COPY ML_Model/ /app/ML_Model/
COPY Property_Testing/ /app/Property_Testing/

# Model assets: each registered model's checkpoint, dataset, and training-latent seed cache.
# s15v1 (15^3):
COPY Archive/d15e200/ /app/Archive/d15e200/
COPY dataset_15.npz /app/dataset_15.npz
COPY Evaluation/latent_data/Freq_FNO_15_training_latent_data.txt /app/Evaluation/latent_data/Freq_FNO_15_training_latent_data.txt
# s10v1 (10^3, old model — pickled against models_epi_3d_10):
COPY Archive/e1200_100/ /app/Archive/e1200_100/
COPY dataset.npz /app/dataset.npz
COPY Evaluation/latent_data/Freq_FNO_training_latent_data.txt /app/Evaluation/latent_data/Freq_FNO_training_latent_data.txt

# The web service
COPY web/backend/main.py web/backend/model_service.py /app/web/backend/

WORKDIR /app/web/backend
EXPOSE 8000

# Cap the math-library threads to 1. On a many-core host, numpy/scipy's BLAS otherwise spawns a thread per core and can DEADLOCK inside the FEM linear solve — observed as /evaluate hanging on sim-hub (idle CPU, request never completing). One thread is ample for this workload and keeps evaluate reliable. Baked in here so it can't be forgotten on `docker run`.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
