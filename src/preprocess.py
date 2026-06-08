import os
import pandas as pd
from pathlib import Path
from src.config import load_config
from src.feast_utils import _s3_storage_options

config = load_config()
RAW_PATH = Path(config["data"]["raw_path"])
DROP_COLUMNS = {"CholCheck", "AnyHealthcare"}
EXPECTED_COLUMNS = [
    "Diabetes_012", "HighBP", "HighChol", "CholCheck", "BMI",
    "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth",
    "DiffWalk", "Sex", "Age", "Education", "Income",
]
VALID_RANGES = {
    "Diabetes_012": (0, 2),
    "HighBP": (0, 1),
    "HighChol": (0, 1),
    "CholCheck": (0, 1),
    "BMI": (10, 100),
    "Smoker": (0, 1),
    "Stroke": (0, 1),
    "HeartDiseaseorAttack": (0, 1),
    "PhysActivity": (0, 1),
    "Fruits": (0, 1),
    "Veggies": (0, 1),
    "HvyAlcoholConsump": (0, 1),
    "AnyHealthcare": (0, 1),
    "NoDocbcCost": (0, 1),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "DiffWalk": (0, 1),
    "Sex": (0, 1),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}
DOMAIN_MISSING_VALUES = {"BMI": 0}

# load data


def load_data(path=None) -> pd.DataFrame:
    if path is None:
        path = os.environ.get("RAW_DATA_PATH") or str(RAW_PATH)

    path_str = str(path)

    if path_str.startswith("s3://"):
        print(f"Loading raw data from S3: {path_str}")
        df = pd.read_csv(path_str, storage_options=_s3_storage_options())
    else:
        print(f"Loading raw data from local path: {path_str}")
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        df = pd.read_csv(p)

    print(f"Loaded data from {path_str}")
    print(f"Shape: {df.shape}")
    return df


# Validation functions
def _format_check(check_name: str, passed: bool, details) -> dict:
    return {"check": check_name, "passed": passed, "details": details}


def check_schema(df: pd.DataFrame) -> dict:
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
    return _format_check("schema", not missing, {"missing_columns": missing, "extra_columns": extra})


def check_nulls(df: pd.DataFrame) -> dict:
    available = [col for col in EXPECTED_COLUMNS if col in df.columns]
    nulls = df[available].isnull().sum()
    nulls = nulls[nulls > 0].to_dict()
    total_nulls = int(sum(nulls.values()))
    return _format_check("null_values", total_nulls == 0, {"total_nulls": total_nulls, "nulls_by_column": nulls})


def check_missing_values(df: pd.DataFrame) -> dict:
    overall_missing = int(df.isnull().sum().sum())
    overall_missing_columns = int((df.isnull().sum() > 0).sum())

    domain_issues = {
        column: {
            "placeholder_missing_value": missing_value,
            "count": int((df[column] == missing_value).sum()),
        }
        for column, missing_value in DOMAIN_MISSING_VALUES.items()
        if column in df.columns and int((df[column] == missing_value).sum()) > 0
    }

    details = {
        "overall_missing_values": overall_missing,
        "overall_missing_columns": overall_missing_columns,
        "domain_specific_issues": domain_issues or "No domain-specific missing values found",
    }
    return _format_check("missing_values", overall_missing == 0 and not domain_issues, details)


def check_ranges(df: pd.DataFrame) -> dict:
    issues = {
        column: {
            "expected_range": [lower, upper],
            "actual_min": df[column].min(),
            "actual_max": df[column].max(),
            "invalid_count": int((~df[column].between(lower, upper)).sum()),
        }
        for column, (lower, upper) in VALID_RANGES.items()
        if column in df.columns and int((~df[column].between(lower, upper)).sum()) > 0
    }
    return _format_check("valid_ranges", not issues, issues or "All values are within valid ranges")


def check_duplicates(df: pd.DataFrame) -> dict:
    duplicate_count = int(df.duplicated().sum())
    total_rows = len(df)
    return _format_check(
        "duplicates",
        duplicate_count == 0,
        {
            "duplicate_rows": duplicate_count,
            "total_rows": total_rows,
            "duplicate_percentage": round((duplicate_count / total_rows) * 100, 2) if total_rows else 0,
        },
    )


def validate_data(df: pd.DataFrame, raise_error: bool = False) -> pd.DataFrame:
    checks = [
        check_schema(df),
        check_nulls(df),
        check_missing_values(df),
        check_ranges(df),
        check_duplicates(df),
    ]

    print("\nDATA VALIDATION REPORT")
    print("-" * 40)
    all_passed = True

    for result in checks:
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"{result['check']:<25}: {status}")
        print(f"Details: {result['details']}\n")
        if not result["passed"]:
            all_passed = False

    if not all_passed and raise_error:
        raise ValueError("Data validation failed.")
    return df


# Cleaning functions
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed duplicate rows: {before - len(df):,}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if "BMI" not in df.columns:
        return df

    df = df.copy()
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["BMI"] = df["BMI"].replace(0, pd.NA).astype("Float64")
    df["BMI"] = df["BMI"].fillna(df["BMI"].median())
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if "BMI" not in df.columns:
        return df

    df = df.copy()
    q1, q3 = df["BMI"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = max(10, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    outliers = ((df["BMI"] < lower_bound) | (df["BMI"] > upper_bound)).sum()
    df["BMI"] = df["BMI"].clip(lower=lower_bound, upper=upper_bound)

    print(f"BMI outliers capped: {outliers}")
    print(f"Bounds used: [{lower_bound:.2f}, {upper_bound:.2f}]")
    return df


def drop_low_information_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [col for col in DROP_COLUMNS if col in df.columns]
    if not existing:
        return df

    df = df.drop(columns=existing)
    print(f"Dropped low-information columns: {existing}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    for step in (remove_duplicates, handle_missing_values, handle_outliers, drop_low_information_columns):
        df = step(df)

    print("Cleaning done")
    print(f"Final cleaned shape: {df.shape}")
    return df
