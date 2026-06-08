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
