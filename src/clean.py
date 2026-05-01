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