import pandas as pd
from pathlib import Path
from src.config import load_config

config = load_config()
RAW_PATH = Path(config["data"]["raw_path"])

def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    print(f"Loaded data from {path}")
    print(f"Shape: {df.shape}")

    return df