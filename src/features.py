import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bins = [0, 18.5, 25, 30, float("inf")]
    labels = ["Underweight", "Normal", "Overweight", "Obese"]

    df["BMI_cat"] = pd.cut(df["BMI"], bins=bins, labels=labels)

    print("BMI category feature created")

    return df
