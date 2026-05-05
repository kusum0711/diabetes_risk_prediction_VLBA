import pandas as pd
from pathlib import Path


def create_feast_data(
    input_path="data/processed/diabetes_processed.csv",
    output_path="data/processed/diabetes_features.parquet",
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    # Always recreate clean patient_id for this static dataset
    if "patient_id" in df.columns:
        df = df.drop(columns=["patient_id"])

    df.insert(0, "patient_id", range(1, len(df) + 1))

    # Feast needs a timestamp column
    if "event_timestamp" in df.columns:
        df = df.drop(columns=["event_timestamp"])

    df["event_timestamp"] = pd.Timestamp("2024-01-01 00:00:00")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print("Feast data created successfully.")
    print(f"Output: {output_path}")
    print(f"Shape: {df.shape}")

    return df


def main():
    create_feast_data()


if __name__ == "__main__":
    main()