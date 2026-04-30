# ==========================================
# MILESTONE 2: COMPLETE PIPELINE
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv(r"C:\Users\max18\OneDrive\Desktop\vlba\diabetes.csv", sep=";")

print("\n" + "="*60)
print("INITIAL DATASET")
print("="*60)
print("Shape before cleaning:", df.shape)

# -------------------------------
# 2. DATA QUALITY CHECK
# -------------------------------

print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
print(df.isnull().sum())

print("\n" + "="*60)
print("DUPLICATE ANALYSIS")
print("="*60)

dup_count = df.duplicated().sum()
dup_pct = (dup_count / len(df)) * 100

print(f"Total duplicates: {dup_count}")
print(f"Percentage: {dup_pct:.2f}%")

print("\nSample duplicate rows:")
print(df[df.duplicated()].head())

# -------------------------------
# 3. DATA CLEANING
# -------------------------------
df = df.drop_duplicates().reset_index(drop=True)

print("\n" + "="*60)
print("AFTER CLEANING")
print("="*60)
print("Shape after removing duplicates:", df.shape)

# -------------------------------
# 4. FEATURE ENGINEERING
# -------------------------------

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

print("\n" + "="*60)
print("BMI CATEGORY DISTRIBUTION")
print("="*60)
print(df['BMI_cat'].value_counts())

# -------------------------------
# 5. TARGET DISTRIBUTION
# -------------------------------
print("\n" + "="*60)
print("TARGET DISTRIBUTION (Diabetes_012)")
print("="*60)

print(df['Diabetes_012'].value_counts())

# Histogram
plt.figure(figsize=(6,4))
df['Diabetes_012'].value_counts().sort_index().plot(kind='bar')
plt.title("Distribution of Diabetes_012")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# -------------------------------
# 6. OUTLIER DETECTION
# -------------------------------
print("\n" + "="*60)
print("OUTLIER DETECTION RESULTS")
print("="*60)

numerical_cols = ['BMI', 'Age', 'MentHlth', 'PhysHlth']

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print("\n" + "-"*40)
    print(f"Feature: {col}")
    print(f"Outliers: {len(outliers)}")
    print(f"Lower bound: {lower:.2f}")
    print(f"Upper bound: {upper:.2f}")

# -------------------------------
# 7. FINAL DATASET CHECK
# -------------------------------
print("\n" + "="*60)
print("FINAL DATASET")
print("="*60)

print("Final shape:", df.shape)
print("Columns:")
print(df.columns)

# -------------------------------
# 8. SAVE CLEANED DATA
# -------------------------------
df.to_csv("diabetes_cleaned.csv", index=False)

print("\nDataset saved as 'diabetes_cleaned.csv'")