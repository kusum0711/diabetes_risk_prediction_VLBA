# """
# Step 5: Push prepared features to storage.
# Maps to: VLBA slides 12-13 (Feature Store).

# Two outputs:
#   1. Parquet file uploaded to S3 (Garage) — permanent storage
#   2. Features registered in Feast — available for training + serving

# Why Feast matters (from VLBA slides):
#   - Training and serving use the same feature definitions
#   - No mismatch between "notebook features" and "production features"
#   - Central registry of what each feature means
#   - Version control for feature data
# """

# import os
# import logging
# import pandas as pd
# import boto3

# logger = logging.getLogger("data-pipeline")


# def save_to_parquet(df: pd.DataFrame, path: str = "/tmp/diabetes_features.parquet"):
#     """Save prepared features as Parquet (Feast preferred format)."""
#     df.to_parquet(path, index=False)
#     logger.info(f"  Saved to {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
#     return path


# def upload_to_s3(
#     local_path: str,
#     bucket: str = "features",
#     key: str = "diabetes/features.parquet",
# ):
#     """Upload prepared features to S3 (Garage on the VM)."""
#     s3 = boto3.client(
#         "s3",
#         endpoint_url=os.getenv("AWS_ENDPOINT_URL", "http://garage:3900"),
#         aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#         aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#         region_name="garage",
#     )

#     s3.upload_file(local_path, bucket, key)
#     logger.info(f"  Uploaded to s3://{bucket}/{key}")


# def push_to_store(df: pd.DataFrame):
#     """
#     Save features and upload to S3.

#     In production with Feast SDK, this would also do:
#         from feast import FeatureStore
#         store = FeatureStore(repo_path="feast_repo/")
#         store.materialize_incremental(end_date=datetime.now())

#     For Milestone II, we upload to S3 where Feast reads from.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Engineered feature set with patient_id and event_timestamp.
#     """
#     logger.info("STORE: saving features")

#     parquet_path = save_to_parquet(df)
#     upload_to_s3(parquet_path)

#     logger.info("STORE: done")

# src/pipeline/store.py

import pandas as pd
from pathlib import Path


def save_data(df: pd.DataFrame, filename: str = "diabetes_processed.csv"):
    output_dir = Path("../../data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename

    df.to_csv(file_path, index=False)

    print(f"Data saved to {file_path}")


def load_processed_data(filename: str = "diabetes_processed.csv") -> pd.DataFrame:
    file_path = Path("../../data/processed") / filename
    return pd.read_csv(file_path)