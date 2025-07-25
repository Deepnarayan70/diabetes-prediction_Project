import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv('diabetes_prediction_impure_data.csv')
df = df.fillna(df.mean(numeric_only=True))

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

if 'FamilyHistory' in df.columns:
    df['FamilyHistory'] = df['FamilyHistory'].map({'Yes': 1, 'No': 0})
df['Diabetic'] = df['Diabetic'].round().astype(int)   

X = df.drop('Diabetic', axis=1)
y = df['Diabetic']




# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# ================================
# Logistic Regression Model
# ================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_preds))
print("Classification Report:\n", classification_report(y_test, lr_preds))


# ================================
# Decision Tree Model
# ================================
dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

print("=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_test, dt_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_preds))
print("Classification Report:\n", classification_report(y_test, dt_preds))




#================================
# Final Comparison
#================================
lr_acc = accuracy_score(y_test, lr_preds)
dt_acc = accuracy_score(y_test, dt_preds)

print("=== Final Comparison ===")
print(f"Logistic Regression Accuracy: {lr_acc:.2f}")
print(f"Decision Tree Accuracy: {dt_acc:.2f}")

if lr_acc > dt_acc:
    print("Logistic Regression performed better.")
elif dt_acc > lr_acc:
    print("Decision Tree performed better.")
else:
    print("Both models performed equally.")


