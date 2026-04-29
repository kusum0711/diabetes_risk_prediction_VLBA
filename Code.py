# ==========================================
# MILESTONE 2: DATA QUALITY + PREPARATION
# COMPLETE PIPELINE CODE
# ==========================================

import pandas as pd

# -------------------------------
# 1. LOAD DATASET
# -------------------------------
df = pd.read_csv("diabetes.csv", sep=";")

print("===== INITIAL DATASET =====")
print("Shape before:", df.shape)

# -------------------------------
# 2. DATA QUALITY CHECK
# -------------------------------

# Missing Values
print("\n===== MISSING VALUES =====")
missing_values = df.isnull().sum()
print(missing_values)
print("Total Missing Values:", missing_values.sum())

# Duplicate Analysis
print("\n===== DUPLICATE ANALYSIS =====")
dup_count = df.duplicated().sum()
dup_pct = (dup_count / len(df)) * 100

print(f"Duplicate rows found: {dup_count} ({dup_pct:.2f}%)")

# Show sample duplicates
print("\nSample duplicate rows:")
print(df[df.duplicated()].head())

# -------------------------------
# 3. DATA CLEANING
# -------------------------------

# Remove duplicates (verified exact matches across all columns)
df = df.drop_duplicates().reset_index(drop=True)

print("\n===== AFTER CLEANING =====")
print("Shape after removing duplicates:", df.shape)

# -------------------------------
# 4. FEATURE ENGINEERING
# -------------------------------

# BMI Category (0–3)
print("\n===== FEATURE ENGINEERING: BMI CATEGORY =====")

def bmi_category(bmi):
    if bmi < 18.5:
        return 0   # Underweight
    elif bmi < 25:
        return 1   # Normal
    elif bmi < 30:
        return 2   # Overweight
    else:
        return 3   # Obese

df['BMI_cat'] = df['BMI'].apply(bmi_category)

# Diabetes Binary
print("\n===== FEATURE ENGINEERING: DIABETES BINARY =====")

# NOTE:
# Original 3-class variable analyzed first,
# then converted to binary based on project goal (risk detection)

df['Diabetes_Binary'] = df['Diabetes_012'].apply(
    lambda x: 0 if x == 0 else 1
)

# Drop original column
df = df.drop(columns=['Diabetes_012'])

# -------------------------------
# 5. FINAL OUTPUT CHECK
# -------------------------------

print("\n===== FINAL DATASET =====")
print("Columns:", df.columns)

print("\nBMI_cat Distribution:")
print(df['BMI_cat'].value_counts())

print("\nDiabetes_Binary Distribution:")
print(df['Diabetes_Binary'].value_counts())

print("\nFinal Shape:", df.shape)

# -------------------------------
# 6. SAVE CLEANED DATA
# -------------------------------

df.to_csv("diabetes_cleaned.csv", index=False)

print("\n===== PROCESS COMPLETE =====")
print("Cleaned dataset saved as 'diabetes_cleaned.csv'")
