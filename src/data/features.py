# """
# Step 4: Feature engineering.
# Maps to: EDA notebook Section 10 (data preparation).

# Creates the model-ready feature set:
#   - Binary target (prediabetes + diabetes combined)
#   - Entity ID for Feast
#   - Event timestamp for Feast
#   - BMI category bins (optional engineered feature)

# Target encoding justification (from EDA):
#   - Prediabetes (class 1) has only 4,631 records (1.8%)
#   - 46:1 imbalance makes 3-class classification impractical
#   - Both prediabetes and diabetes require clinical intervention
#   - Combined binary target has 15.8% positive rate
# """

# import logging
# import pandas as pd
# import numpy as np
# from datetime import datetime

# logger = logging.getLogger("data-pipeline")


# def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Combine prediabetes (1) and diabetes (2) into single at-risk class.

#     Before: Diabetes_012 = 0, 1, 2
#     After:  target = 0 (healthy) or 1 (at-risk)
#     """
#     df["target"] = (df["Diabetes_012"] >= 1).astype(int)
#     df = df.drop(columns=["Diabetes_012"])

#     pos_rate = df["target"].mean() * 100
#     logger.info(f"  Binary target created: {pos_rate:.1f}% positive (at-risk)")

#     return df


# def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Create BMI category feature based on WHO classifications.

#     Underweight (<18.5) = 0, Normal (18.5-25) = 1,
#     Overweight (25-30) = 2, Obese (30+) = 3

#     EDA showed diabetes rate rises from 6.2% (Normal) to 31.2% (Obese III),
#     so discrete bins may capture non-linear BMI effects.
#     """
#     bins = [0, 18.5, 25, 30, 100]
#     labels = [0, 1, 2, 3]
#     df["BMI_cat"] = pd.cut(df["BMI"], bins=bins, labels=labels).astype(int)
#     logger.info(f"  BMI_cat created: {df['BMI_cat'].value_counts().to_dict()}")

#     return df


# def add_feast_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add columns required by Feast feature store.

#     Feast needs:
#       - entity column (unique ID per row)
#       - event_timestamp (when this data point was created)
#     """
#     df["patient_id"] = range(len(df))
#     df["event_timestamp"] = datetime.now()

#     return df


# def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Apply all feature engineering steps.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Cleaned dataset.

#     Returns
#     -------
#     pd.DataFrame
#         Model-ready dataset with engineered features.
#     """
# src/pipeline/features.py

import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bins = [0, 18.5, 25, 30, float("inf")]
    labels = ["Underweight", "Normal", "Overweight", "Obese"]

    df["BMI_cat"] = pd.cut(df["BMI"], bins=bins, labels=labels)

    print("BMI category feature created")

    return df
