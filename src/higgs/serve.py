"""
serve.py: FastAPI service for the Higgs MLP.

Run:  uvicorn serve:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import os
from pathlib import Path
from typing import Annotated, Literal

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .model import load_checkpoint, load_scaler

# serve.py -> higgs/ -> src/ -> repo root
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("HIGGS_ARTIFACTS", ROOT / "artifacts"))

# Loaded once, when the server process starts (not on every request)
MODEL = load_checkpoint("artifacts/model.pt")
MEAN, STD = load_scaler("artifacts/scaler.json")
N_FEATURES = len(MEAN)

# A float that rejects NaN and inf, since training dropped rows with missing values
Feature = Annotated[float, Field(allow_inf_nan=False)]


class PredictRequest(BaseModel):
    features: list[Feature] = Field(min_length=N_FEATURES, max_length=N_FEATURES)


class PredictResponse(BaseModel):
    label: Literal["signal", "background"]
    probability: float  # P(signal)


app = FastAPI(title="Higgs classifier")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    x = torch.tensor(req.features, dtype=torch.float32)
    x = ((x - MEAN) / STD).unsqueeze(0)  # scale, then add a batch dimension
    with torch.no_grad():
        prob = torch.sigmoid(MODEL(x)).item()  # logit -> P(signal)
    return PredictResponse(
        label="signal" if prob >= 0.5 else "background",
        probability=round(prob, 4),
    )
