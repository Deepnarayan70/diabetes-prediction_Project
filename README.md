# Diabetes Prediction using Machine Learning

This project is aimed at predicting whether a person is diabetic or not based on various health parameters using machine learning techniques. The workflow includes data cleaning, exploratory data analysis (EDA), model building, and evaluation.

## Project Structure

- `1_data_cleaning.py` – Cleans the dataset by handling missing values, type corrections, and saves the processed version.
- `2_eda.py` – Performs Exploratory Data Analysis to understand data distribution and correlations between features.
- `3_diabetes_model_training.py` – Trains and evaluates classification models (Logistic Regression and Decision Tree).

## Dataset Information

- **Source**: `diabetes_prediction_impure_data.csv`
- **Description**: Contains medical and health records like glucose levels, BMI, insulin, blood pressure, etc., along with a label indicating diabetic status (`diabetes` column).

## Project Goals

- Understand and clean impure health data
- Explore relationships and patterns in data using EDA
- Build and evaluate machine learning models to predict diabetes

## ML Models Used

- **Logistic Regression**
- **Decision Tree Classifier**

Both models were trained using scikit-learn and evaluated using metrics like accuracy, confusion matrix, and classification report.

## Outcomes

- Cleaned and processed real-world health dataset
- Identified significant features through EDA
- Built predictive models with decent accuracy and interpretability

## Tools & Libraries

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

## Future Scope

- Add performance visualizations (confusion matrix, ROC curve, etc.)
- Improve model using Random Forest or XGBoost
- Web deployment using Streamlit or Flask


