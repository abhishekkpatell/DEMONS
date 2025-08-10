# service.py
import os
from typing import List, Optional, Dict

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

APP_TITLE = "Job Role Classifier API"
APP_VERSION = "1.0.0"

# ---- Config via env (override if needed) ----
BASE_DIR = os.getenv("WORK_DIR", os.getcwd())
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "job_role_classifier.joblib"))
LE_PATH    = os.getenv("LE_PATH",    os.path.join(BASE_DIR, "label_encoder.joblib"))
EMBEDDER_NAME = os.getenv("EMBEDDER_NAME", "all-MiniLM-L6-v2")

# ---- FastAPI app ----
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# ---- Globals (loaded on startup) ----
clf = None
le = None
embedder: SentenceTransformer | None = None

# ---- Schemas ----
class PredictItem(BaseModel):
    text: str

class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    predictions: List[str]

class ProbaResponse(BaseModel):
    predictions: List[str]
    probabilities: List[Dict[str, float]]  # per item: {"LabelA": 0.12, "LabelB": 0.34, ...}

class HealthResponse(BaseModel):
    status: str
    model_path: str
    labels: List[str]
    embedder: str

def _safe_load():
    global clf, le, embedder
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        raise FileNotFoundError(
            f"Artifacts missing. MODEL_PATH={MODEL_PATH} exists={os.path.exists(MODEL_PATH)}, "
            f"LE_PATH={LE_PATH} exists={os.path.exists(LE_PATH)}"
        )
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    embedder = SentenceTransformer(EMBEDDER_NAME)

@app.on_event("startup")
def on_startup():
    _safe_load()

def _embed(texts: List[str]) -> np.ndarray:
    assert embedder is not None
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embs)

# ---- Routes ----
@app.get("/health", response_model=HealthResponse)
def health():
    if clf is None or le is None or embedder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_path=MODEL_PATH,
        labels=list(le.classes_),
        embedder=EMBEDDER_NAME
    )

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if clf is None or le is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.texts:
        return PredictResponse(predictions=[])
    X = _embed(req.texts)
    y = clf.predict(X)
    labels = le.inverse_transform(y).tolist()
    return PredictResponse(predictions=labels)

@app.post("/predict_proba", response_model=ProbaResponse)
def predict_proba(req: PredictRequest):
    """Returns per-class probabilities (if supported).
       For LogisticRegression this is available. """
    if clf is None or le is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not hasattr(clf, "predict_proba"):
        raise HTTPException(status_code=400, detail="Classifier does not support predict_proba")
    if not req.texts:
        return ProbaResponse(predictions=[], probabilities=[])
    X = _embed(req.texts)
    probs = clf.predict_proba(X)  # shape [n, C]
    y = np.argmax(probs, axis=1)
    labels = le.inverse_transform(y).tolist()
    classes = le.classes_.tolist()
    prob_maps = [{cls: float(p[i]) for i, cls in enumerate(classes)} for p in probs]
    return ProbaResponse(predictions=labels, probabilities=prob_maps)

@app.post("/reload")
def reload_model():
    _safe_load()
    return {"status": "reloaded", "model_path": MODEL_PATH, "embedder": EMBEDDER_NAME}
