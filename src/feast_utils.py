import os
import json
import subprocess
import urllib.request
import pandas as pd

from pathlib import Path
from datetime import datetime

from feast import FeatureStore


FEATURE_SERVICE_NAME = "diabetes_feature_service"

FEATURES = [
    "diabetes_features:HighBP",
    "diabetes_features:HighChol",
    "diabetes_features:BMI",
    "diabetes_features:Smoker",
    "diabetes_features:Stroke",
    "diabetes_features:HeartDiseaseorAttack",
    "diabetes_features:PhysActivity",
    "diabetes_features:Fruits",
    "diabetes_features:Veggies",
    "diabetes_features:HvyAlcoholConsump",
    "diabetes_features:NoDocbcCost",
    "diabetes_features:GenHlth",
    "diabetes_features:MentHlth",
    "diabetes_features:PhysHlth",
    "diabetes_features:DiffWalk",
    "diabetes_features:Sex",
    "diabetes_features:Age",
    "diabetes_features:Education",
    "diabetes_features:Income",
    "diabetes_features:age_metabolic_risk",
    "diabetes_features:HealthRiskScore",
    # "diabetes_features:metabolic_risk",
]

# Where feature data lives when running against S3/Garage (server profile).
FEAST_SOURCE_S3 = "s3://feast/diabetes_features.parquet"


# ---------------------------------------------------
# ENVIRONMENT HELPERS (local file store vs S3/Garage server profile)
# ---------------------------------------------------

def _s3_endpoint():
    return os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT")


def _use_s3():
    """Server profile is active when an S3 endpoint is configured."""
    return bool(_s3_endpoint())


def _s3_storage_options():
    """storage_options for pandas/s3fs against a custom S3 endpoint (MinIO/Garage)."""
    return {
        "key": os.environ.get("AWS_ACCESS_KEY_ID"),
        "secret": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {"endpoint_url": _s3_endpoint()},
    }


# ---------------------------------------------------
# CONFIGURE FEAST (local file store vs S3/Postgres server profile)
# ---------------------------------------------------

def configure_feast(repo_path="feature_repo"):
    """Select the Feast profile based on the environment.

    When an S3 endpoint is configured (AWS_ENDPOINT_URL / S3_ENDPOINT), activate
    the server profile (registry + offline data on S3, online store on Postgres)
    by copying feature_store.yaml.server over feature_store.yaml, and point the
    feature definitions at the S3 parquet. Otherwise leave the committed local
    file/sqlite config untouched so local dev keeps working.
    """

    repo_path = Path(repo_path)

    # Local registry/sqlite live under feature_repo/data (relative paths in the
    # feature_store.yaml); make sure the dir exists on host and in containers.
    (repo_path / "data").mkdir(parents=True, exist_ok=True)

    if not _use_s3():
        print("  S3 not configured — using local feature_store.yaml")
        return None

    server_yaml = repo_path / "feature_store.yaml.server"
    target = repo_path / "feature_store.yaml"
    target.write_text(server_yaml.read_text())

    # Make `feast apply` (subprocess) and FeatureStore read the S3 source.
    os.environ.setdefault("FEAST_SOURCE_PATH", FEAST_SOURCE_S3)

    print("  Using server Feast profile (S3 registry + offline store)")
    return target


# ---------------------------------------------------
# CREATE FEAST DATA
# ---------------------------------------------------

def create_feast_data(
    input_path="data/processed/diabetes_processed.csv",
    output_path="data/processed/diabetes_features.parquet",
):

    input_path = Path(input_path)

    df = pd.read_csv(input_path)

    # recreate patient_id
    if "patient_id" in df.columns:
        df = df.drop(columns=["patient_id"])

    df.insert(0, "patient_id", range(1, len(df) + 1))

    # create timestamp for Feast
    if "event_timestamp" in df.columns:
        df = df.drop(columns=["event_timestamp"])

    df["event_timestamp"] = pd.Timestamp(
        "2024-01-01 00:00:00"
    )

    if _use_s3():
        df.to_parquet(
            FEAST_SOURCE_S3,
            index=False,
            storage_options=_s3_storage_options(),
        )
        print(f"  Feast feature data written to {FEAST_SOURCE_S3}")
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print("  Feast feature data created")

    return df


# ---------------------------------------------------
# APPLY FEAST
# ---------------------------------------------------

def apply_feast(repo_path="feature_repo"):

    configure_feast(repo_path)

    print("Applying Feast definitions...")

    subprocess.run(
        ["feast", "apply"],
        cwd=repo_path,
        check=True,
    )

    print("  Feast apply completed")


# ---------------------------------------------------
# GET TRAINING DATA (offline / historical features)
# ---------------------------------------------------

def get_training_data(
    source_path="data/processed/diabetes_features.parquet",
    output_path="data/processed/training_data_from_feast.csv",
    repo_path="feature_repo",
):

    configure_feast(repo_path)

    output_path = Path(output_path)

    if _use_s3():
        df = pd.read_parquet(
            FEAST_SOURCE_S3,
            storage_options=_s3_storage_options(),
        )
    else:
        df = pd.read_parquet(Path(source_path))

    entity_df = df[
        ["patient_id", "event_timestamp"]
    ]

    store = FeatureStore(repo_path=repo_path)

    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=store.get_feature_service(FEATURE_SERVICE_NAME),
    ).to_df()

    # add target column
    target_df = df[
        ["patient_id", "Diabetes_012"]
    ]

    feature_df = feature_df.merge(
        target_df,
        on="patient_id",
        how="left",
    )

    feature_df = (
        feature_df
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(output_path, index=False)

    print("  Training data retrieved from Feast")
    print(f"Shape: {feature_df.shape}")

    return feature_df


# ---------------------------------------------------
# FEATURE STORE MODE (online via Feast REST server vs offline local access)
# ---------------------------------------------------
# FEATURE_STORE_MODE selects how features are read and how the online store is
# materialized:
#   online  -> Feast REST server (FEAST_URL): /get-online-features for reads,
#              /materialize-incremental for materialization.
#   offline -> read the feature parquet directly (in-memory), and materialize
#              into the local online store via the embedded Feast SDK.

FEATURE_COLS = [f.split(":", 1)[1] for f in FEATURES]

# Feast feature server (REST). Endpoints are appended to FEAST_URL, so on the
# cluster set FEAST_URL to include any mount prefix; locally `feast serve` is at
# the root (http://localhost:6000).
FEAST_URL = os.environ.get("FEAST_URL", "http://localhost:6000")


def feature_store_mode():
    """Return the active feature store mode: 'online' or 'offline' (default).

    online  -> use the Feast online store via the REST server (FEAST_URL).
    offline -> read the offline feature parquet directly.
    """
    return os.environ.get("FEATURE_STORE_MODE", "online").lower()


# ---------------------------------------------------
# OFFLINE: read the feature parquet directly
# ---------------------------------------------------

def read_features_parquet(
    local_path="data/processed/diabetes_features.parquet",
):
    """Read the full feature parquet (S3 when configured, else local file).

    Returns a DataFrame indexed by patient_id.
    """
    if _use_s3():
        df = pd.read_parquet(
            FEAST_SOURCE_S3,
            storage_options=_s3_storage_options(),
        )
    else:
        df = pd.read_parquet(Path(local_path))

    if "patient_id" not in df.columns:
        raise RuntimeError("Feature parquet has no 'patient_id' column")

    return df.set_index("patient_id")


# ---------------------------------------------------
# ONLINE: Feast REST server (FEAST_URL)
# ---------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars/arrays that json.dumps can't handle natively."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _post(path, payload):
    """POST JSON to the Feast feature server and return the parsed response."""
    url = FEAST_URL.rstrip("/") + path
    data = json.dumps(payload, cls=_NumpyEncoder).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def materialize_incremental():
    """Materialize features into the online store via the Feast REST server."""
    end_ts = datetime.utcnow().replace(microsecond=0).isoformat()
    print(f"  POST {FEAST_URL}/materialize-incremental (end_ts={end_ts})")
    _post("/materialize-incremental", {"end_ts": end_ts})
    print("  Online store materialized (via Feast REST server)")


def push_to_online_store_rest(
    local_path="data/processed/diabetes_features.parquet",
    feature_view_name="diabetes_features",
    batch_size=5000,
):
    """Write feature rows directly to the Feast online store via /write-to-online-store.

    Reads the feature parquet (S3 when configured, else local), then POSTs
    column-oriented batches to the Feast feature server. Timestamps are
    serialised as ISO-8601 strings so they survive JSON round-tripping.
    """
    if _use_s3():
        df = pd.read_parquet(FEAST_SOURCE_S3, storage_options=_s3_storage_options())
    else:
        df = pd.read_parquet(Path(local_path))

    # Feast requires event_timestamp; serialise to ISO string for JSON.
    if "event_timestamp" in df.columns:
        df = df.copy()
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"]).dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

    total = len(df)
    batches = range(0, total, batch_size)
    print(
        f"  Pushing {total} rows to online store in {len(batches)} batch(es) "
        f"(feature_view={feature_view_name!r}, batch_size={batch_size})"
    )

    for i, start in enumerate(batches):
        chunk = df.iloc[start: start + batch_size]
        payload = {
            "feature_view_name": feature_view_name,
            "df": chunk.to_dict(orient="list"),
            "allow_registry_cache": True,
            "transform_on_write": False,
        }
        _post("/write-to-online-store", payload)
        print(f"    batch {i + 1}/{len(batches)}: rows {start}–{start + len(chunk) - 1} pushed")

    print(f"  Online store push complete ({total} rows)")


def get_online_features_rest(patient_ids):
    """Fetch online features for patient ids via the Feast REST server.

    Uses the registered FeatureService so the REST server can resolve the
    request against the registry. Returns a DataFrame with a patient_id column
    plus the feature columns.
    """
    payload = {
        "feature_service": FEATURE_SERVICE_NAME,
        "entities": {"patient_id": [int(pid) for pid in patient_ids]},
        "full_feature_names": False,
    }
    resp = _post("/get-online-features", payload)

    feature_names = resp["metadata"]["feature_names"]
    results = resp["results"]
    columns = {
        feature_names[i]: results[i]["values"]
        for i in range(len(feature_names))
    }
    return pd.DataFrame(columns)


# ---------------------------------------------------
# ONLINE STORE via embedded SDK (used to materialize in OFFLINE mode)
# ---------------------------------------------------

def get_feature_store(repo_path="feature_repo", apply=False):
    """Return a FeatureStore for the active profile (local vs S3/server).

    When apply=True, run `feast apply` first so the registry exists before the
    store is opened (needed before the first SDK materialize).
    """
    configure_feast(repo_path)

    if apply:
        subprocess.run(["feast", "apply"], cwd=repo_path, check=True)

    return FeatureStore(repo_path=repo_path)


def get_online_features_sdk(patient_ids, repo_path="feature_repo"):
    """Fetch features from the online store via the Feast SDK.

    Bypasses the Feast REST server — reads directly from the configured
    online store (sqlite locally, Postgres on the cluster).
    """
    store = get_feature_store(repo_path)
    df = store.get_online_features(
        features=store.get_feature_service(FEATURE_SERVICE_NAME),
        entity_rows=[{"patient_id": int(pid)} for pid in patient_ids],
    ).to_df()
    return df.set_index("patient_id")


def materialize_online_store(repo_path="feature_repo", end=None):
    """Populate the local online store from the parquet via the Feast SDK.

    Runs `feast teardown` then `feast apply` then materialize_incremental.
    Teardown is required in CI: each stage runs in a fresh container while the
    S3 registry is shared.  Without it, `feast apply` sees "No changes to
    infrastructure" (registry already up-to-date from prepare-data) and returns
    early without calling update_infra() — so SQLite tables are never created.
    Teardown clears the S3 registry and drops any existing tables (IF EXISTS),
    so the subsequent apply sees all objects as new and always creates the tables.
    """
    configure_feast(repo_path)
    subprocess.run(["feast", "teardown"], cwd=repo_path, check=False)

    store = get_feature_store(repo_path, apply=True)

    end = end or datetime.utcnow()
    print(f"  Materializing online store (end={end.isoformat()}) ...")
    store.materialize_incremental(end_date=end)
    print("  Online store materialized (via Feast SDK)")

    return store


# ---------------------------------------------------
# MODE-AWARE DISPATCHERS
# ---------------------------------------------------

def materialize(mode=None):
    """Materialize the online store using the method for the running mode."""
    mode = mode or feature_store_mode()
    if mode == "online":
        materialize_incremental()          # REST → Feast server's online store
    else:
        materialize_online_store()         # SDK → local online store


def push_features(mode=None):
    """Push feature rows directly to the online store for the running mode.

    online  -> /write-to-online-store REST endpoint
    offline -> SDK materialize_incremental (local sqlite)
    """
    mode = mode or feature_store_mode()
    if mode == "online":
        push_to_online_store_rest()
    else:
        materialize_online_store()


def fetch_features(patient_ids, mode=None):
    """Return a patient_id-indexed feature frame for the running mode.

    online  -> Feast REST /get-online-features
    offline -> Feast SDK get_online_features (reads from configured online store)
    """
    mode = mode or feature_store_mode()
    if mode == "online":
        return get_online_features_rest(patient_ids).set_index("patient_id")
    return get_online_features_sdk(patient_ids)
