# """
# Step 1: Load raw data from S3 (Garage).
# Maps to: EDA notebook Section 1 (data loading).

# Instead of pd.read_csv("diabetes.csv") from a local file,
# this reads from S3-compatible storage so the pipeline works
# on any machine — not just your laptop.
# """

# import os
# import logging
# import pandas as pd


# logger = logging.getLogger("data-pipeline")


# def get_s3_client():
#     """Create an S3 client configured for Garage on the VM."""
#     import boto3
#     return boto3.client(
#         "s3",
#         endpoint_url=os.getenv("AWS_ENDPOINT_URL", "http://garage:3900"),
#         aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#         aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#         region_name=os.getenv("AWS_REGION", "garage"),
#     )


# def load_data(bucket: str = "data", key: str = "diabetes.csv") -> pd.DataFrame:
#     """
#     Download CSV from S3 and return as DataFrame.

#     Parameters
#     ----------
#     bucket : str
#         S3 bucket name where the raw data is stored.
#     key : str
#         File path within the bucket.

#     Returns
#     -------
#     pd.DataFrame
#         Raw dataset as loaded — no transformations applied.
#     """
#     s3 = get_s3_client()
#     local_path = "/tmp/diabetes_raw.csv"

#     s3.download_file(bucket, key, local_path)
#     df = pd.read_csv(local_path)

#     logger.info(f"LOAD: {df.shape[0]:,} rows x {df.shape[1]} columns from s3://{bucket}/{key}")

#     return df


# src/pipeline/load.py

import pandas as pd
from pathlib import Path

RAW_PATH = Path("../../data/raw/diabetes.csv")


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    print(f"Loaded data from {path}")
    print(f"Shape: {df.shape}")

    return df