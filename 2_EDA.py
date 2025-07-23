################-------------------------------------------------------EDA-----------------------------------------------------####################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

sns.set(style="whitegrid")

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

#Target Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Diabetic', data=df)
plt.title("Distribution of Diabetic vs Non-Diabetic")
plt.xlabel("Diabetic (1 = Yes, 0 = No)" if df['Diabetic'].dtype != 'int' else "Diabetic")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


#Correlation Heatmap

if df['Diabetic'].dtype == 'O':
    df['Diabetic'] = df['Diabetic'].map({'Yes': 1, 'No': 0})

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


#Boxplots for Outlier Detection

features = ["Age", "BMI", "Glucose", "BloodPressure"]
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Diabetic', y=feature, data=df)
    plt.title(f"{feature} vs Diabetic")
    plt.tight_layout()
    plt.show()


#Pairplot for Visualizing Relationships

sns.pairplot(df, hue="Diabetic", vars=features, corner=True)
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()


#Distribution Plots for Each Feature

for feature in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=feature, kde=True, hue="Diabetic", bins=30, palette='Set1')
    plt.title(f"Distribution of {feature} by Diabetic Status")
    plt.tight_layout()
    plt.show()


#Count Plots for Categorical Features

categorical_features = df.select_dtypes(include='object').columns
for cat in categorical_features:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=cat, hue="Diabetic")
    plt.title(f"{cat} vs Diabetic")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#Print Value Counts for Categorical Columns
for cat in categorical_features:
    print(f"\n--- Value Counts for {cat} ---")
    print(df[cat].value_counts())
