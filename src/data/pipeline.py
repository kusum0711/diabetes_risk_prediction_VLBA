# """
# Data Pipeline Orchestrator
# ==========================
# Runs all data preparation steps in sequence.

# This is the entry point called by GitLab CI/CD.
# Each step is a separate module — this file just calls them in order.

# Usage:
#     python -m src.data.pipeline                    # default: read from S3
#     python -m src.data.pipeline --local data/diabetes.csv  # local file (testing)

# Pipeline flow:
#     load → validate → clean → engineer → store → metrics
# """

# import os
# import sys
# import logging
# import argparse
# import pandas as pd

# from src.data.load import load_data
# from src.data.validate import validate_data
# from src.data.clean import clean_data
# from src.data.features import engineer_features
# from src.data.store import push_to_store
# from src.data.metrics import push_metrics

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     datefmt="%H:%M:%S",
# )
# logger = logging.getLogger("data-pipeline")


# def run_pipeline(local_path: str = None):
#     """
#     Execute the full data pipeline.

#     Parameters
#     ----------
#     local_path : str, optional
#         Path to a local CSV file. If None, reads from S3.
#         Use for local testing without VM access.
#     """
#     logger.info("=" * 55)
#     logger.info("DATA PIPELINE — START")
#     logger.info("=" * 55)

#     # ── Step 1: Load ──
#     if local_path:
#         logger.info(f"LOAD: reading local file {local_path}")
#         df = pd.read_csv(local_path)
#         logger.info(f"LOAD: {df.shape[0]:,} rows x {df.shape[1]} columns")
#     else:
#         df = load_data()

#     # ── Step 2: Validate ──
#     report = validate_data(df)

#     # ── Step 3: Clean ──
#     df = clean_data(df)

#     # ── Step 4: Engineer features ──
#     df = engineer_features(df)

#     # ── Step 5: Push to store ──
#     if local_path:
#         # Local mode — save to disk instead of S3
#         output = "data/prepared_features.parquet"
#         os.makedirs("data", exist_ok=True)
#         df.to_parquet(output, index=False)
#         logger.info(f"STORE: saved locally to {output}")
#     else:
#         push_to_store(df)

#     # ── Step 6: Push metrics ──
#     if not local_path:
#         push_metrics(df, report)
#     else:
#         logger.info("METRICS: skipped (local mode)")

#     # ── Summary ──
#     logger.info("=" * 55)
#     logger.info("DATA PIPELINE — COMPLETE")
#     logger.info(f"  Final shape   : {df.shape}")
#     logger.info(f"  Features      : {[c for c in df.columns if c not in ['target','patient_id','event_timestamp']]}")
#     logger.info(f"  Target rate   : {df['target'].mean()*100:.1f}% at-risk")
#     logger.info("=" * 55)

#     return df


# def main():
#     parser = argparse.ArgumentParser(description="Diabetes data pipeline")
#     parser.add_argument(
#         "--local",
#         type=str,
#         default=None,
#         help="Path to local CSV (skips S3 + Feast + Pushgateway)",
#     )
#     args = parser.parse_args()

#     run_pipeline(local_path=args.local)


# if __name__ == "__main__":
#     main()



# src/pipeline/pipeline.py

from load import load_data
from validate import validate_data
from clean import clean_data
from features import engineer_features
from store import save_data


def run_pipeline():
    df = load_data()

    df = validate_data(df, raise_error=False)

    df = clean_data(df)

    df = engineer_features(df)

    save_data(df)

    print("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()
