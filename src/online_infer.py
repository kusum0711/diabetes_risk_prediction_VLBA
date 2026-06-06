import os
import json
from datetime import datetime

import urllib.error
import urllib.request

import pandas as pd
import mlflow

from src.feast_utils import FEATURES


# Feast feature server (REST). Endpoints are appended to FEAST_URL, so on the
# cluster set FEAST_URL to include any mount prefix (the server spec mounts the
# API under "/feast"); locally `feast serve` is at the root.
FEAST_URL = os.environ.get("FEAST_URL", "http://localhost:6000")

# Short feature column names (without the "diabetes_features:" view prefix).
FEATURE_COLS = [f.split(":", 1)[1] for f in FEATURES]

MODEL_NAME = os.environ.get("MODEL_NAME", "diabetes_random_forest")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "latest")


def _post(path, payload):
    """POST JSON to the Feast feature server and return the parsed response."""
    url = FEAST_URL.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def materialize_incremental():
    """Materialize features into the online store via the Feast server."""
    end_ts = datetime.utcnow().replace(microsecond=0).isoformat()
    print(f"  POST {FEAST_URL}/materialize-incremental (end_ts={end_ts})")
    _post("/materialize-incremental", {"end_ts": end_ts})
    print("  Online store materialized (via Feast server)")


def get_online_features(patient_ids):
    """Fetch online features for patient ids via the Feast server REST API."""
    payload = {
        "entities": {"patient_id": [int(pid) for pid in patient_ids]},
        "features": FEATURES,
        "full_feature_names": False,
    }
    resp = _post("/get-online-features", payload)

    # Response: metadata.feature_names aligned with results[i].values.
    feature_names = resp["metadata"]["feature_names"]
    results = resp["results"]
    columns = {
        feature_names[i]: results[i]["values"]
        for i in range(len(feature_names))
    }
    return pd.DataFrame(columns)


def run_online_infer(patient_ids=None):
    """Low-latency serving via the Feast server: materialize → fetch → predict."""

    print("\n🚀 DIABETES RISK PREDICTION — ONLINE INFERENCE PIPELINE")

    if patient_ids is None:
        patient_ids = [1, 2, 3, 4, 5]

    # ----------------------------------
    # STEP 1: MATERIALIZE → ONLINE STORE (via Feast server)
    # ----------------------------------
    materialize_incremental()

    # ----------------------------------
    # STEP 2: FETCH ONLINE FEATURES (via Feast server)
    # NOTE: /get-online-features is not accessible on the cluster's shared Feast
    # server (project "mlpipeline"). Skipping until a project-scoped server is
    # available.
    # ----------------------------------
    # features = get_online_features(patient_ids)
    # print("\n  Online features:")
    # print(features)

    print("  Online feature retrieval skipped (HTTP route not accessible on cluster)")
    return None


if __name__ == "__main__":
    run_online_infer()
