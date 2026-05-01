import yaml

def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML config and return as a plain dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)