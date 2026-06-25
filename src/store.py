import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from pathlib import Path
from src.config import load_config

config = load_config()
PROCESSED_PATH = Path(config["data"]["processed_path"])


def save_data(df: pd.DataFrame, filename: str = "diabetes_processed.csv"):
    output_dir = PROCESSED_PATH.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename

    df.to_csv(file_path, index=False)

    print(f"Data saved to {file_path}")


def load_processed_data(filename: str = "diabetes_processed.csv") -> pd.DataFrame:
    file_path = PROCESSED_PATH
    return pd.read_csv(file_path)


def load_registered_model(model_name: str, model_version: str):
    """Resolve and load a registered MLflow model. Returns (model, resolved_version)."""
    if model_version and model_version.lower() != "latest":
        uri = f"models:/{model_name}/{model_version}"
        resolved_version = model_version
    else:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise RuntimeError(f"No registered versions found for model '{model_name}'")
        latest = max(versions, key=lambda v: int(v.version))
        uri = f"models:/{model_name}/{latest.version}"
        resolved_version = latest.version

    try:
        model = mlflow.sklearn.load_model(uri)
    except Exception:
        model = mlflow.xgboost.load_model(uri)

    return model, resolved_version
