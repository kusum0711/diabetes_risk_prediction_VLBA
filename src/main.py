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
    push_features,
)

from src.train import run_training
from src.online_infer import run_online_infer
from src.config import load_config
from src.env import API_HOST, API_PORT


# Hand-off file produced by the prepare-data pipeline and consumed by the
# train pipeline. Must match get_training_data's default output_path.
TRAINING_DATA_PATH = "data/processed/training_data_from_feast.csv"


def run_prepare_data():

    print("\n🚀 DIABETES RISK PREDICTION — PREPARE DATA PIPELINE")

    df = load_data()
    df = validate_data(df, raise_error=False)
    df = clean_data(df)
    df = engineer_features(df)
    save_data(df)
    create_feast_data()
    apply_feast()
    training_df = get_training_data()

    print("\n  PREPARE DATA PIPELINE COMPLETED")

    return training_df


def run_train():

    print("\n🚀 DIABETES RISK PREDICTION — TRAIN PIPELINE")

    config = load_config()
    training_df = pd.read_csv(TRAINING_DATA_PATH)
    results = run_training(training_df, config)

    print("\n  TRAIN PIPELINE COMPLETED")

    return results


def run_pipeline():
    """Run the full pipeline (prepare data + train) sequentially."""

    run_prepare_data()
    return run_train()


def run_api():
    """Serve the prediction API (FastAPI via uvicorn).

    Host/port come from API_HOST / API_PORT (default 0.0.0.0:8000). Imported
    lazily so the other stages don't require uvicorn/fastapi.
    """
    import uvicorn

    host = API_HOST
    port = API_PORT

    print(f"\n🚀 DIABETES RISK PREDICTION — API on {host}:{port}")
    uvicorn.run("src.api:app", host=host, port=port)


if __name__ == "__main__":

    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    if stage == "prepare":
        run_prepare_data()
    elif stage == "train":
        run_train()
    elif stage in ("online", "online-infer", "infer"):
        run_online_infer()
    elif stage in ("api", "serve"):
        run_api()
    elif stage == "push":
        push_features()
    else:
        run_pipeline()
