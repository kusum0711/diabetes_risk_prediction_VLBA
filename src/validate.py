import pandas as pd

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


def check_schema(df: pd.DataFrame) -> dict:
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    return {
        "check": "schema",
        "passed": len(missing_columns) == 0,
        "details": {
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
        },
    }


def check_nulls(df: pd.DataFrame) -> dict:
    available_columns = [col for col in EXPECTED_COLUMNS if col in df.columns]

    nulls_by_column = df[available_columns].isnull().sum()
    nulls_by_column = nulls_by_column[nulls_by_column > 0].to_dict()

    total_nulls = sum(nulls_by_column.values())

    return {
        "check": "null_values",
        "passed": total_nulls == 0,
        "details": {
            "total_nulls": int(total_nulls),
            "nulls_by_column": nulls_by_column,
        },
    }


def check_missing_values(df: pd.DataFrame) -> dict:
    """
    Checks domain-specific missing values.

    In this dataset, actual nulls may not exist,
    but invalid placeholder values such as BMI = 0
    should be treated as missing/invalid.
    """

    missing_rules = {
        "BMI": 0,
    }

    issues = {}

    for column, missing_value in missing_rules.items():
        if column in df.columns:
            count = int((df[column] == missing_value).sum())
            if count > 0:
                issues[column] = {
                    "placeholder_missing_value": missing_value,
                    "count": count,
                }

    return {
        "check": "domain_missing_values",
        "passed": len(issues) == 0,
        "details": issues if issues else "No domain-specific missing values found",
    }


def check_ranges(df: pd.DataFrame) -> dict:
    issues = {}

    for column, (lower, upper) in VALID_RANGES.items():
        if column not in df.columns:
            continue

        invalid_mask = ~df[column].between(lower, upper)
        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            issues[column] = {
                "expected_range": [lower, upper],
                "actual_min": df[column].min(),
                "actual_max": df[column].max(),
                "invalid_count": invalid_count,
            }

    return {
        "check": "valid_ranges",
        "passed": len(issues) == 0,
        "details": issues if issues else "All values are within valid ranges",
    }


def check_duplicates(df: pd.DataFrame) -> dict:
    duplicate_count = int(df.duplicated().sum())

    return {
        "check": "duplicates",
        "passed": duplicate_count == 0,
        "details": {
            "duplicate_rows": duplicate_count,
            "total_rows": len(df),
            "duplicate_percentage": round((duplicate_count / len(df)) * 100, 2),
        },
    }


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