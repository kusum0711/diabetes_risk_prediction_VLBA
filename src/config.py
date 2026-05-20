import os
import yaml
from pathlib import Path


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

    # Override MLflow tracking URI from environment variable if set
    # This allows Docker to use http://mlflow:5000 
    # while local uses http://localhost:5000
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        config["mlflow"]["tracking_uri"] = env_uri
        print(f"  MLflow URI overridden from environment: {env_uri}")

    print(f"  Config loaded from: {path}")
    return config


if __name__ == "__main__":
    config = load_config()
    print(config)