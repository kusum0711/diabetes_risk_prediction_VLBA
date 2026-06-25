import os

# ---------------------------------------------------------------------------
# Active — read by the application code
# ---------------------------------------------------------------------------

# MLflow
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

# Model serving
MODEL_NAME = os.environ.get("MODEL_NAME", "diabetes_random_forest")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "latest")

# API server
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# Monitoring
PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "").strip()

# Feast
FEAST_URL = os.environ.get("FEAST_URL", "http://localhost:6000")
FEATURE_STORE_MODE = os.environ.get("FEATURE_STORE_MODE", "online").lower()

# AWS / S3 (used by feast_utils and preprocess for custom S3-compatible endpoints)
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Data
RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH")

# ---------------------------------------------------------------------------
# Available but not read directly — set to configure underlying SDKs or
# enable optional features. Uncomment and use if needed.
# ---------------------------------------------------------------------------

# AWS region — consumed implicitly by boto3/s3fs; not read by application code.
# AWS_REGION         = os.environ.get("AWS_REGION")
# AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION")

# Alternative S3 endpoint key used by some AWS SDK versions and MLflow artifact store.
# AWS_ENDPOINT_URL_S3    = os.environ.get("AWS_ENDPOINT_URL_S3")
# MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL")

# Feast source parquet path override. Handled via os.environ.setdefault() in
# feast_utils.py rather than a module-level constant, so it must remain there.
# FEAST_SOURCE_PATH = os.environ.get("FEAST_SOURCE_PATH")

# Postgres online store — re-enable the postgres profile in feature_store.yaml.server
# and uncomment these when switching from SQLite to a Postgres-backed online store.
# POSTGRES_FEAST_HOST     = os.environ.get("POSTGRES_FEAST_HOST")
# POSTGRES_FEAST_PORT     = os.environ.get("POSTGRES_FEAST_PORT", "5432")
# POSTGRES_FEAST_USER     = os.environ.get("POSTGRES_FEAST_USER")
# POSTGRES_FEAST_PASSWORD = os.environ.get("POSTGRES_FEAST_PASSWORD")
# POSTGRES_FEAST_DB       = os.environ.get("POSTGRES_FEAST_DB")
