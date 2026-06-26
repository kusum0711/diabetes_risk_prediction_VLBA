"""Online-store materialization stage.

Materializes features into the online store, verifies the store is readable,
then runs a sample prediction to confirm the full pipeline end-to-end.
"""

import time

import mlflow

from src.feast_utils import materialize, fetch_features, FEATURE_COLS
from src.config import load_config
from src.store import load_registered_model
from src.env import MODEL_NAME, MODEL_VERSION

_SAMPLE_PATIENT_ID = 1
_REGISTRY_REFRESH_TIMEOUT = 60
_REGISTRY_POLL_INTERVAL = 10


def run_online_infer():
    """Materialize the online store, verify features, then run a sample prediction."""

    print("\nDIABETES RISK PREDICTION — MATERIALIZE ONLINE STORE")

    materialize()

    print(f"\n  Verifying online store for patient_id={_SAMPLE_PATIENT_ID} ...")
    deadline = time.time() + _REGISTRY_REFRESH_TIMEOUT
    features = None
    while True:
        try:
            features = fetch_features([_SAMPLE_PATIENT_ID])
            null_count = features[FEATURE_COLS].isna().all(axis=1).sum()
            if null_count > 0:
                print(f"  WARNING: patient_id={_SAMPLE_PATIENT_ID} returned all-null features — was materialization successful?")
            else:
                print(f"  Online store returned {len(FEATURE_COLS)} feature(s) — OK.")
            break
        except Exception as exc:
            if "404" in str(exc) and time.time() < deadline:
                print(f"  Registry not yet refreshed ({exc}), retrying in {_REGISTRY_POLL_INTERVAL}s ...")
                time.sleep(_REGISTRY_POLL_INTERVAL)
            else:
                print(f"  WARNING: feature fetch failed ({exc}).")
                break

    if features is not None and not features[FEATURE_COLS].isna().all(axis=1).any():
        print(f"\n  Sample prediction for patient_id={_SAMPLE_PATIENT_ID} ...")
        try:
            config = load_config()
            mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
            threshold = float(config.get("model", {}).get("class_1_threshold", 0.5))
            model, _ = load_registered_model(MODEL_NAME, MODEL_VERSION)
            X = features.loc[[_SAMPLE_PATIENT_ID], FEATURE_COLS].astype("float32")
            proba = float(model.predict_proba(X)[:, 1][0])
            label = "Diabetes/At risk" if proba >= threshold else "No diabetes"
            print(f"  → {label} (probability={proba:.4f}, threshold={threshold})")
        except Exception as exc:
            print(f"  WARNING: sample prediction failed ({exc}).")
    else:
        print("\n  Skipping sample prediction — features unavailable.")

    print("\n  Online store ready. Predictions are served by the prediction API.")


if __name__ == "__main__":
    run_online_infer()
