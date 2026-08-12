"""
main.py — the FastAPI service the frontend talks to.

Three endpoints matching web/frontend/src/api/types.ts exactly:
  GET  /models    -> [{id, label, gridDim}]
  POST /generate  -> {voxels, filledTarget, fa, epsE}
  POST /evaluate  -> {achieved}
Plus GET /health for liveness checks.

The model is loaded ONCE (warm in memory) — see model_service.py. This process must stay alive (it is not serverless): generate() runs live autograd, evaluate() runs FEM.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import model_service as ms

# Which frontend origin(s) may call us
_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# Warm the default model at startup so the first real request isn't cold.
_DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", ms.MODELS[0].id if ms.MODELS else "")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if _DEFAULT_MODEL:
        ms.get_service(_DEFAULT_MODEL) # load into memory
    yield

app = FastAPI(title="Metamaterial Inverse-Design API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- request bodies (responses are plain dicts from model_service) ---
class GenerateRequest(BaseModel):
    # 22 entries
    target: list[float | None] = Field(..., min_length=ms.CHANNEL_COUNT, max_length=ms.CHANNEL_COUNT)
    model: str

class EvaluateRequest(BaseModel):
    voxels: list[int]

# --- endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def get_models():
    return ms.list_models()

@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        svc = ms.get_service(req.model)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown model id: {req.model!r}")
    try:
        return svc.generate(req.target)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    try:
        return ms.evaluate(req.voxels)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
