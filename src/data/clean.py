# """
# Step 3: Clean data based on EDA findings.
# Maps to: EDA notebook Section 9 (outlier treatment)
#        + EDA notebook Section 10 (feature dropping).

# Every decision here was justified by actual analysis:
#   - CholCheck dropped  → 96.3% constant (EDA Section 6)
#   - AnyHealthcare dropped → 95.1% constant (EDA Section 6)
#   - BMI capped at 45   → 98th percentile; BMI=98 likely error (EDA Section 9)
#   - Outliers in MentHlth/PhysHlth kept → valid chronic illness data (EDA Section 9)
# """

# import logging
# import pandas as pd

# logger = logging.getLogger("data-pipeline")

# # Features to drop — near-constant, low predictive value
# DROP_COLUMNS = ["CholCheck", "AnyHealthcare"]

# # BMI cap — 98th percentile from EDA
# BMI_CAP = 45


# def drop_weak_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Remove features with near-zero variance.

#     CholCheck: 96.3% = 1, correlation with target = +0.068
#     AnyHealthcare: 95.1% = 1, correlation with target = +0.015
#     """
#     dropped = [c for c in DROP_COLUMNS if c in df.columns]
#     df = df.drop(columns=dropped, errors="ignore")
#     logger.info(f"  Dropped {len(dropped)} columns: {dropped}")
#     return df


# def cap_bmi(df: pd.DataFrame, cap: float = BMI_CAP) -> pd.DataFrame:
#     """
#     Cap BMI at 98th percentile.

#     BMI outliers above 41.5 have 36.5% diabetes rate (vs 14% normal),
#     so they carry signal — we cap rather than remove.
#     BMI = 98 is likely a data entry error (world record is ~95).
#     """
#     n_before = (df["BMI"] > cap).sum()
#     df["BMI"] = df["BMI"].clip(upper=cap)
#     logger.info(f"  BMI capped at {cap}: {n_before:,} values clipped")
#     return df


# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Apply all cleaning steps.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Validated raw dataset.

#     Returns
#     -------
#     pd.DataFrame
#         Cleaned dataset ready for feature engineering.
#     """
#     df = df.copy()

#     logger.info("CLEAN: applying transformations")
#     df = drop_weak_features(df)
#     df = cap_bmi(df)

#     logger.info(f"CLEAN: done — shape {df.shape}")
#     return df

# src/pipeline/clean.py

import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)

    print(f"Removed duplicate rows: {before - after:,}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # BMI = 0 is not realistic, treat as missing
    if "BMI" in df.columns:
        df["BMI"] = df["BMI"].replace(0, pd.NA)
        df["BMI"] = df["BMI"].fillna(df["BMI"].median())

    return df



def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "BMI" in df.columns:
        # Calculate IQR-based bounds
        q1 = df["BMI"].quantile(0.25)
        q3 = df["BMI"].quantile(0.75)
        iqr = q3 - q1

        lower_bound = max(10, q1 - 1.5 * iqr)   # keep realistic minimum
        upper_bound = q3 + 1.5 * iqr

        before_outliers = ((df["BMI"] < lower_bound) | (df["BMI"] > upper_bound)).sum()

        # Apply capping
        df["BMI"] = df["BMI"].clip(lower=lower_bound, upper=upper_bound)

        print(f"BMI outliers capped: {before_outliers}")
        print(f"Bounds used: [{lower_bound:.2f}, {upper_bound:.2f}]")

    return df


def drop_low_information_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = ["CholCheck", "AnyHealthcare"]
    existing_cols = [col for col in drop_cols if col in df.columns]

    df = df.drop(columns=existing_cols)

    if existing_cols:
        print(f"Dropped low-information columns: {existing_cols}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Remove duplicate rows first
    df = remove_duplicates(df)

    # 2. Handle missing / invalid values
    df = handle_missing_values(df)

    # 3. Handle outliers
    df = handle_outliers(df)

    # 4. Drop low-information columns
    df = drop_low_information_columns(df)

    print("Cleaning done")
    print(f"Final cleaned shape: {df.shape}")

    return df