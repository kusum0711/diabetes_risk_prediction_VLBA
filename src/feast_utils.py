import os
import subprocess
import pandas as pd

from pathlib import Path

from feast import FeatureStore


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
        features=FEATURES,
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
