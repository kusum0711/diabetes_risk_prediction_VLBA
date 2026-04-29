# ==========================================
# MILESTONE 2: FULL DATA QUALITY + PREPARATION
# ==========================================

import pandas as pd
import numpy as np

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("diabetes.csv", sep=";")

print("===== INITIAL DATASET =====")
print("Shape:", df.shape)

# -------------------------------
# 2. DATA QUALITY CHECK
# -------------------------------

# 2.1 Missing Values
print("\n===== MISSING VALUES =====")
missing_values = df.isnull().sum()
print(missing_values)
print("Total Missing:", missing_values.sum())

# 2.2 Duplicate Records
print("\n===== DUPLICATES =====")
dup_count = df.duplicated().sum()
dup_pct = (dup_count / len(df)) * 100

print(f"Duplicates: {dup_count} ({dup_pct:.2f}%)")

# Show sample duplicates
print("\nSample duplicate rows:")
print(df[df.duplicated()].head())

# 2.3 Data Types
print("\n===== DATA TYPES =====")
print(df.dtypes)

# 2.4 Invalid Values Check
print("\n===== INVALID VALUES CHECK =====")

# Example checks
print("Negative BMI values:", (df['BMI'] < 0).sum())
print("Invalid Diabetes values:", df['Diabetes_012'].isin([0,1,2]).value_counts())

# 2.5 Outlier Detection (IQR method for BMI)
print("\n===== OUTLIER DETECTION (BMI) =====")

Q1 = df['BMI'].quantile(0.25)
Q3 = df['BMI'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['BMI'] < lower_bound) | (df['BMI'] > upper_bound)]
print("Number of BMI outliers:", len(outliers))

# 2.6 Class Imbalance (Original target)
print("\n===== CLASS DISTRIBUTION (3-CLASS) =====")
print(df['Diabetes_012'].value_counts(normalize=True) * 100)

# -------------------------------
# 3. DATA CLEANING
# -------------------------------

print("\n===== DATA CLEANING =====")

# Remove duplicates (verified exact matches)
df = df.drop_duplicates().reset_index(drop=True)

print("Shape after duplicate removal:", df.shape)

# -------------------------------
# 4. FEATURE ENGINEERING
# -------------------------------

# 4.1 BMI Category
print("\n===== BMI CATEGORY =====")

def bmi_category(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    else:
        return 3

df['BMI_cat'] = df['BMI'].apply(bmi_category)

print(df['BMI_cat'].value_counts())

# 4.2 Binary Target (after analysis)
print("\n===== BINARY TARGET CREATION =====")

df['Diabetes_Binary'] = df['Diabetes_012'].apply(
    lambda x: 0 if x == 0 else 1
)

print(df['Diabetes_Binary'].value_counts(normalize=True) * 100)

# Drop original targetu
df = df.drop(columns=['Diabetes_012'])

# -------------------------------
# 5. FINAL DATASET CHECK
# -------------------------------

print("\n===== FINAL DATASET =====")

print("Final Shape:", df.shape)
print("Columns:", df.columns)

print("\nBMI_cat Distribution:")
print(df['BMI_cat'].value_counts())

print("\nDiabetes_Binary Distribution:")
print(df['Diabetes_Binary'].value_counts())

# -------------------------------
# 6. SAVE CLEANED DATA
# -------------------------------

df.to_csv("diabetes_cleaned.csv", index=False)

print("\n===== PROCESS COMPLETE =====")
print("Dataset ready for modeling")
