import subprocess

from src.preprocess import load_data
from src.preprocess import validate_data
from src.preprocess import clean_data
from src.features import engineer_features
from src.store import save_data
from src.create_feast_data import create_feast_data
from src.get_training_data import get_training_data_from_feast


def apply_feast():
    print("Applying Feast feature definitions...")

    subprocess.run(
        ["feast", "apply"],
        cwd="feature_repo",
        check=True,
    )

    print("Feast apply completed successfully.")


def run_pipeline():
    print("Starting preprocessing pipeline...")

    df = load_data()

    df = validate_data(df, raise_error=False)

    df = clean_data(df)

    df = engineer_features(df)

    save_data(df)

    print("Processed data saved successfully.")

    print("Creating Feast-compatible Parquet file...")
    create_feast_data()

    apply_feast()

    print("Creating training data from Feast...")
    get_training_data_from_feast()

    print("Full pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()