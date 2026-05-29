import pandas as pd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



# FEATURE_GROUPS = {

#     "baseline": [],

#     "bmi_features": [

#         "BMI_cat_Obese",
#         "BMI_cat_Overweight",
#         "BMI_cat_Underweight",
#     ],

#     "health_features": [
#         "Total_Unhealthy_Days",
#     ],

#     "age_features": [

#         "Age_cat_Old",
#         "Age_cat_Young",
#     ],

#     "interaction_features": [

#         "BMI_Age",
#         "BMI_Unhealthy",
#         "Health_Age",
#     ],
# }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # # =====================================================
    # # BMI Categories
    # # =====================================================

    # bmi_bins = [0, 18.5, 25, 30, float("inf")]

    # bmi_labels = [
    #     "Underweight",
    #     "Normal",
    #     "Overweight",
    #     "Obese",
    # ]

    # df["BMI_cat"] = pd.cut(
    #     df["BMI"],
    #     bins=bmi_bins,
    #     labels=bmi_labels,
    # )

    # print("  BMI category feature created")

    # # =====================================================
    # # Total Unhealthy Days
    # # =====================================================

    # df["Total_Unhealthy_Days"] = (
    #     df["MentHlth"] +
    #     df["PhysHlth"]
    # )

    # print("  Total_Unhealthy_Days feature created")

    # # =====================================================
    # # Age Categories
    # # =====================================================

    # age_bins = [0, 4, 8, float("inf")]

    # age_labels = [
    #     "Young",
    #     "Middle_Age",
    #     "Old",
    # ]

    # df["Age_cat"] = pd.cut(
    #     df["Age"],
    #     bins=age_bins,
    #     labels=age_labels,
    # )

    # print("  Age category feature created")


    # # =====================================================
    # # Interaction Features
    # # =====================================================

    # df["BMI_Age"] = (
    #     df["BMI"] * df["Age"]
    # )

    # df["BMI_Unhealthy"] = (
    #     df["BMI"] * df["Total_Unhealthy_Days"]
    # )

    # df["Health_Age"] = (
    #     df["GenHlth"] * df["Age"]
    # )

    # print("  Interaction features created")
    # df["BMI_scaled"] = scaler.fit_transform(df[["BMI"]])

#     df["metabolic_risk"] = (
#         (df["BMI"] >= 30).astype(int) +
#         df["HighBP"] +
#         df["HighChol"]
#     )

#     print("  Metabolic risk feature created")
    
#     df["health_burden"] = (
#         df["GenHlth"] +
#         (df["PhysHlth"] / 10) +
#         (df["MentHlth"] / 10)
#     )
#     print("  Health burden feature created")
   

#     df["age_metabolic_risk"] = (
#     df["Age"] * df["metabolic_risk"]
# )
#     print("  Age-metabolic risk interaction feature created")

    return df
# def build_feature_set(
#     X,
#     selected_groups,
# ):

#     selected_features = []

#     for group in selected_groups:

#         selected_features.extend(
#             FEATURE_GROUPS[group]
#         )

#     engineered_features = sum(
#         FEATURE_GROUPS.values(),
#         []
#     )

#     base_features = [

#         c for c in X.columns

#         if c not in engineered_features
#     ]

#     final_features = (
#         base_features +
#         selected_features
#     )

#     available_features = [

#         f for f in final_features

#         if f in X.columns
#     ]

#     missing_features = [

#         f for f in final_features

#         if f not in X.columns
#     ]

#     if missing_features:

#         print(
#             f"Missing features:"
#             f" {missing_features}"
#         )

#     return X[available_features]