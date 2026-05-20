from src.preprocess import (
    load_data,
    validate_data,
    clean_data,
)

from src.features import engineer_features
from src.store import save_data

from src.feast_utils import (
    create_feast_data,
    apply_feast,
    get_training_data,
)

from src.train import run_training
from src.config import load_config


def run_pipeline():

    print("\n🚀 DIABETES RISK PREDICTION PIPELINE")

    config = load_config()

    # ----------------------------------
    # STEP 1: LOAD DATA
    # ----------------------------------
    df = load_data()

    # ----------------------------------
    # STEP 2: VALIDATE DATA
    # ----------------------------------
    df = validate_data(
        df,
        raise_error=False,
    )

    # ----------------------------------
    # STEP 3: CLEAN DATA
    # ----------------------------------
    df = clean_data(df)

    # ----------------------------------
    # STEP 4: FEATURE ENGINEERING
    # ----------------------------------
    df = engineer_features(df)
    save_data(df)

    # ----------------------------------
    # STEP 5: SAVE FEAST DATA
    # ----------------------------------
    create_feast_data()

    # ----------------------------------
    # STEP 6: APPLY FEAST
    # ----------------------------------
    apply_feast()

    # ----------------------------------
    # STEP 7: GET TRAINING DATA
    # ----------------------------------
    training_df = get_training_data()

    # ----------------------------------
    # STEP 8: TRAIN MODELS
    # ----------------------------------
    results = run_training(
        training_df,
        config,
    )

    print("\n  PIPELINE COMPLETED")

    return results


if __name__ == "__main__":
    run_pipeline()