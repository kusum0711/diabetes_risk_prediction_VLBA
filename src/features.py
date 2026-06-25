import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["BMI_binned"] = (df["BMI"] >= 30).astype(int)

    df["HealthRiskScore"] = (
        df["GenHlth"] +
        df["HighBP"] +
        df["BMI_binned"]
    )
    print("  HealthRiskScore feature created")

    df["metabolic_risk"] = (
        df["BMI_binned"] +
        df["HighBP"] +
        df["HighChol"]
    )
    df["age_metabolic_risk"] = (
        df["Age"] * df["metabolic_risk"]
    )
    print("  Age-metabolic risk interaction feature created")

    return df
