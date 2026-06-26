import yaml
from pathlib import Path
from src.env import MLFLOW_TRACKING_URI


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    base_dir = Path(__file__).resolve().parent.parent
    path = Path(config_path)
    if not path.is_absolute():
        path = base_dir / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Normalize data paths so relative paths work regardless of current working directory.
    def resolve_data_path(path_value: str) -> str:
        if not path_value:
            return path_value
        p = Path(path_value)
        return str(p if p.is_absolute() else base_dir / p)

    if "data" in config:
        for key in ["raw_path", "processed_path", "training_data_path"]:
            if key in config["data"]:
                config["data"][key] = resolve_data_path(config["data"][key])

    if MLFLOW_TRACKING_URI:
        config["mlflow"]["tracking_uri"] = MLFLOW_TRACKING_URI
        print(f"  MLflow URI overridden from environment: {MLFLOW_TRACKING_URI}")

    print(f"  Config loaded from: {path}")
    return config


if __name__ == "__main__":
    config = load_config()
    print(config)
