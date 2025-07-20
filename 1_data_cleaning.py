#############    Purpose : Data loading, missing value handling      #############
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Load the dataset
df = pd.read_csv("diabetes_prediction_impure_data.csv")

# Function to convert string values to NaN if they can't be converted to float
def to_numeric_or_nan(val):
    try:
        return float(val)
    except:
        return pd.NA

# Convert applicable columns to numeric, replacing invalid strings with NaN
for col in ["Age", "BMI", "Glucose", "BloodPressure"]:
    df[col] = df[col].apply(to_numeric_or_nan)

df = df.apply(pd.to_numeric, errors='ignore')

df = df[df["Diabetic"].notna()]

# Impute missing numeric values with median
imputer = SimpleImputer(strategy="median")
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])


print("Cleaned dataset:")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
