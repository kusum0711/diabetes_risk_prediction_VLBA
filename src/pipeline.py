from src.load import load_data
from src.validate import validate_data
from src.clean import clean_data
from src.features import engineer_features
from src.store import save_data


def run_pipeline():
    df = load_data()

    df = validate_data(df, raise_error=False)

    df = clean_data(df)

    df = engineer_features(df)

    save_data(df)

    print("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()
