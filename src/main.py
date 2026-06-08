import sys

import pandas as pd

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
from src.online_infer import run_online_infer
from src.config import load_config


# Hand-off file produced by the prepare-data pipeline and consumed by the
# train pipeline. Must match get_training_data's default output_path.
TRAINING_DATA_PATH = "data/processed/training_data_from_feast.csv"


def run_prepare_data():

    print("\n🚀 DIABETES RISK PREDICTION — PREPARE DATA PIPELINE")

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

    print("\n  PREPARE DATA PIPELINE COMPLETED")

    return training_df


def run_train():

    print("\n🚀 DIABETES RISK PREDICTION — TRAIN PIPELINE")

    config = load_config()

    # ----------------------------------
    # STEP 8: LOAD TRAINING DATA
    # (produced by the prepare-data pipeline)
    # ----------------------------------
    training_df = pd.read_csv(TRAINING_DATA_PATH)

    # ----------------------------------
    # STEP 9: TRAIN MODELS
    # ----------------------------------
    results = run_training(
        training_df,
        config,
    )

    print("\n  TRAIN PIPELINE COMPLETED")

    return results


def run_pipeline():
    """Run the full pipeline (prepare data + train) sequentially."""

    run_prepare_data()
    return run_train()


if __name__ == "__main__":

    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    if stage == "prepare":
        run_prepare_data()
    elif stage == "train":
        run_train()
    elif stage in ("online", "online-infer", "infer"):
        run_online_infer()
    else:
        run_pipeline()
