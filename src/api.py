"""Prediction API.

Serves diabetes-risk predictions for patients by patient_id. The model is the
one trained by the train pipeline and registered in MLflow.

Where features come from is controlled by FEATURE_STORE_MODE (see src.feast_utils):
    online  -> Feast REST server (FEAST_URL): /get-online-features
    offline -> the feature parquet directly, cached in memory

Flow per request:
    patient_id(s) -> fetch feature row(s) (mode-aware) -> load registered MLflow
    model -> predict probability + class (configured threshold) -> JSON.

Run locally (inside the app image, which has the deps):
    python -m src.main api      # or: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import os

import mlflow

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.feast_utils import (
    FEATURE_COLS,
    feature_store_mode,
    read_features_parquet,
    fetch_features,
    materialize as feast_materialize,
)
from src.config import load_config

TARGET_COLUMN = "Diabetes_012"

# Which registered model to serve. MODEL_VERSION may be a version number
# ("2") or the literal "latest" to resolve the highest registered version.
MODEL_NAME = os.environ.get("MODEL_NAME", "diabetes_random_forest")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "latest")


# ---------------------------------------------------------------------------
# Lazy-loaded state (model + features), cached after first load.
# ---------------------------------------------------------------------------
class _State:
    model = None
    model_version = None
    features = None  # offline DataFrame indexed by patient_id (offline mode / listing)
    threshold = 0.5


_state = _State()


def _resolve_model_uri():
    """Return (uri, version) for the model to load."""
    if MODEL_VERSION and MODEL_VERSION.lower() != "latest":
        return f"models:/{MODEL_NAME}/{MODEL_VERSION}", MODEL_VERSION

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for model '{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{MODEL_NAME}/{latest.version}", latest.version


def _load_model_obj(uri):
    """Load the registered model with predict_proba support.

    XGBoost models are logged via the xgboost flavor; everything else via the
    sklearn flavor. Both flavors restore an estimator exposing predict_proba.
    """
    if "xgboost" in MODEL_NAME.lower():
        return mlflow.xgboost.load_model(uri)
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception:
        return mlflow.xgboost.load_model(uri)


def _ensure_offline_features():
    """Load + cache the offline parquet (offline mode and /patients listing)."""
    if _state.features is None:
        _state.features = read_features_parquet()
    return _state.features


def _ensure_model():
    """Load the registered model + threshold once, then reuse it."""
    if _state.model is None:
        config = load_config()
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        _state.threshold = float(
            config.get("model", {}).get("class_1_threshold", 0.5)
        )
        uri, version = _resolve_model_uri()
        _state.model = _load_model_obj(uri)
        _state.model_version = version


def _fetch_features(patient_ids):
    """Return a patient_id-indexed feature frame for the given ids.

    Both online and offline modes read from the online store (REST or SDK).
    Raises 404 for ids not present or with all-null features (not materialized).
    """
    df = fetch_features(patient_ids)
    missing = [
        pid for pid in patient_ids
        if pid not in df.index or df.loc[pid, FEATURE_COLS].isna().all()
    ]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No features for patient_id(s): {missing}. "
                "Has the online store been materialized?"
            ),
        )
    return df


def _predict_patients(patient_ids):
    """Predict for a list of patient ids."""
    _ensure_model()
    features = _fetch_features(patient_ids)

    rows = features.loc[patient_ids]
    X = rows[FEATURE_COLS].astype("float32")

    proba = _state.model.predict_proba(X)[:, 1]
    threshold = _state.threshold

    results = []
    for pid, p in zip(patient_ids, proba):
        pred = int(p >= threshold)
        results.append(
            {
                "patient_id": int(pid),
                "prediction": pred,
                "label": "Diabetes/At risk" if pred else "No diabetes",
                "probability": round(float(p), 4),
                "threshold": threshold,
            }
        )
    return results


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Diabetes Risk Prediction API",
    description="Predict diabetes risk for patients by patient_id.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    patient_ids: list[int]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/patients")
def list_patients(limit: int = 50):
    """List available patient ids (from the feature parquet)."""
    features = _ensure_offline_features()
    ids = features.index.tolist()
    return {
        "total": len(ids),
        "patient_ids": [int(i) for i in ids[:limit]],
    }


@app.get("/model")
def model_info():
    """Report which model/version is being served."""
    _ensure_model()
    return {
        "model_name": MODEL_NAME,
        "model_version": _state.model_version,
        "feature_store_mode": feature_store_mode(),
        "threshold": _state.threshold,
        "features": FEATURE_COLS,
    }


@app.get("/predict/{patient_id}")
def predict_one(patient_id: int):
    """Predict diabetes risk for a single patient."""
    result = _predict_patients([patient_id])[0]
    result["model"] = MODEL_NAME
    result["model_version"] = _state.model_version
    return result


@app.post("/predict")
def predict_many(req: PredictRequest):
    """Predict diabetes risk for a batch of patients."""
    if not req.patient_ids:
        raise HTTPException(status_code=400, detail="patient_ids must not be empty")
    results = _predict_patients(req.patient_ids)
    return {
        "model": MODEL_NAME,
        "model_version": _state.model_version,
        "predictions": results,
    }


@app.post("/materialize")
def materialize():
    """Populate the online store, using the method for the running mode.

    online  -> Feast REST server /materialize-incremental
    offline -> local online store via the embedded Feast SDK

    Run after training/prepare (or whenever the parquet changes).
    """
    feast_materialize()
    return {"status": "materialized", "feature_store_mode": feature_store_mode()}


@app.post("/reload")
def reload_state():
    """Drop cached model + features so the next request reloads them."""
    _state.model = None
    _state.features = None
    return {"status": "reloaded"}
