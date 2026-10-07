import os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils import clean

MODEL_PATH = "model/sentiment_model.joblib"
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request
    state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Customer Feedback Classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    confident: bool
    threshold: float
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Same cleaning function as training
    text = clean(req.text)
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty after cleaning")

    model = state["model"]
    proba = model.predict_proba([text])[0]
    idx = int(proba.argmax())
    confidence = float(proba[idx])

    return PredictResponse(
        label=str(model.classes_[idx]),
        confidence=round(confidence, 4),
        confident=confidence >= THRESHOLD,
        threshold=THRESHOLD,
        probabilities={
            str(c): round(float(p), 4) for c, p in zip(model.classes_, proba)
        },
    )
