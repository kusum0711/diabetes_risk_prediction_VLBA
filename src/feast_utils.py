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
    # "diabetes_features:health_burden",
    # "diabetes_features:age_metabolic_risk",
]


# ---------------------------------------------------
# CREATE FEAST DATA
# ---------------------------------------------------

def create_feast_data(
    input_path="data/processed/diabetes_processed.csv",
    output_path="data/processed/diabetes_features.parquet",
):

    input_path = Path(input_path)
    output_path = Path(output_path)

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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(output_path, index=False)

    print("  Feast feature data created")

    return df


# ---------------------------------------------------
# APPLY FEAST
# ---------------------------------------------------

def apply_feast(repo_path="feature_repo"):

    print("Applying Feast definitions...")

    subprocess.run(
        ["feast", "apply"],
        cwd=repo_path,
        check=True,
    )

    print("  Feast apply completed")


# ---------------------------------------------------
# GET TRAINING DATA
# ---------------------------------------------------

def get_training_data(
    source_path="data/processed/diabetes_features.parquet",
    output_path="data/processed/training_data_from_feast.csv",
    repo_path="feature_repo",
):

    source_path = Path(source_path)
    output_path = Path(output_path)

    df = pd.read_parquet(source_path)

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