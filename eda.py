# -*- coding: utf-8 -*-
"""
Created on Thu May 29 23:07:40 2025

@author: munna
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# 1. Load dataset
file_path = r"C:\embro_quality\DATA_SET\data without infertility _final.csv"
df = pd.read_csv(file_path)

# 2. Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# 3. Create AFC column (sum of Follicle No. (L) and Follicle No. (R))
df["AFC"] = df["Follicle No. (L)"] + df["Follicle No. (R)"]



# 4. Clean AMH column (convert to numeric, handle commas and spaces)
df["AMH(ng/mL)"] = pd.to_numeric(df["AMH(ng/mL)"].astype(str).str.replace(",", "").str.strip(), errors="coerce")

# 5. Handle missing values
# Drop 'Unnamed: 42' as it has too many missing values and likely irrelevant
if "Unnamed: 42" in df.columns:
    df = df.drop(columns=["Unnamed: 42"])

# Impute missing values for Marraige Status (Yrs) with median
if df["Marraige Status (Yrs)"].isnull().sum() > 0:
    df["Marraige Status (Yrs)"].fillna(df["Marraige Status (Yrs)"].median(), inplace=True)

# Impute missing values for AMH(ng/mL) with median
if df["AMH(ng/mL)"].isnull().sum() > 0:
    df["AMH(ng/mL)"].fillna(df["AMH(ng/mL)"].median(), inplace=True)

# Impute missing values for Fast food (Y/N) with mode
if df["Fast food (Y/N)"].isnull().sum() > 0:
    df["Fast food (Y/N)"].fillna(df["Fast food (Y/N)"].mode()[0], inplace=True)

# 6. Features to analyze
features = ["FSH(mIU/mL)", "LH(mIU/mL)", "Age (yrs)", "AMH(ng/mL)", "BMI", "AFC"]

# 7. Basic info
print("Dataset shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary statistics:\n", df[features].describe())

# 8. Missing values check after imputation
print("\nMissing values after imputation:\n", df.isnull().sum())

# 9. Distributions (Histograms)
df[features].hist(bins=30, figsize=(12, 10))
plt.suptitle("Distributions of Features", fontsize=16)
plt.tight_layout()
plt.show()

# 10. Boxplots for outlier detection
for col in features:
    plt.figure(figsize=(8, 1.5))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot - {col}")
    plt.show()

# 11. Correlation matrix
correlation = df[features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 12. Skewness check
skewness = df[features].skew().sort_values(ascending=False)
print("\nSkewness of Features:\n", skewness)

# Optional: 13. Pairplot (can be slow on large data, uncomment if needed)
# sns.pairplot(df[features])
# plt.show()
df.to_csv(r"C:\embro_quality\DATA_SET\data_after_EDA.csv", index=False)
print("Final cleaned dataset saved after EDA.")
