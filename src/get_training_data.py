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
    "diabetes_features:BMI_cat",
]


def get_training_data_from_feast(
    source_path="data/processed/diabetes_features.parquet",
    output_path="data/processed/training_data_from_feast.csv",
    repo_path="feature_repo",
):
    source_path = Path(source_path)
    output_path = Path(output_path)

    df = pd.read_parquet(source_path)

    entity_df = df[["patient_id", "event_timestamp"]]

    store = FeatureStore(repo_path=repo_path)

    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURES,
    ).to_df()

    # Safer: merge target using patient_id instead of assuming row order
    target_df = df[["patient_id", "Diabetes_012"]]

    feature_df = feature_df.merge(
        target_df,
        on="patient_id",
        how="left",
    )

    # Optional: sort by patient_id for cleaner output
    feature_df = feature_df.sort_values("patient_id").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)

    print("Training data created from Feast successfully.")
    print(f"Output: {output_path}")
    print(f"Shape: {feature_df.shape}")
    print(feature_df.head())

    return feature_df