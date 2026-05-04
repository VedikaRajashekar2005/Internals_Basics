from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import numpy as np
import json
import os
import requests

app = FastAPI()

# Load model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)


class BuildInput(BaseModel):
    test_count: int = Field(..., ge=50, le=1000)
    artifact_size_mb: float = Field(..., ge=10, le=500)
    parallel_jobs: int = Field(..., ge=1, le=8)
    is_full_rebuild: int = Field(..., ge=0, le=1)


@app.get("/heartbeat")
def heartbeat():
    return {"alive": True, "service": "PipelineIQ build_time_min API"}


@app.post("/predict")
def predict(data: BuildInput):
    features = np.array([[
        data.test_count,
        data.artifact_size_mb,
        data.parallel_jobs,
        data.is_full_rebuild
    ]])
    prediction = model.predict(features)[0]
    return {"prediction": round(float(prediction), 4)}