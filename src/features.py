import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # =====================================================
    # BMI Categories
    # =====================================================

    bmi_bins = [0, 18.5, 25, 30, float("inf")]

    bmi_labels = [
        "Underweight",
        "Normal",
        "Overweight",
        "Obese",
    ]

    df["BMI_cat"] = pd.cut(
        df["BMI"],
        bins=bmi_bins,
        labels=bmi_labels,
    )

    print("✅ BMI category feature created")

    # =====================================================
    # Total Unhealthy Days
    # =====================================================

    df["Total_Unhealthy_Days"] = (
        df["MentHlth"] +
        df["PhysHlth"]
    )

    print("✅ Total_Unhealthy_Days feature created")

    # =====================================================
    # Age Categories
    # =====================================================

    age_bins = [0, 4, 8, float("inf")]

    age_labels = [
        "Young",
        "Middle_Age",
        "Old",
    ]

    df["Age_cat"] = pd.cut(
        df["Age"],
        bins=age_bins,
        labels=age_labels,
    )

    print("✅ Age category feature created")

    return df