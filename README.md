# Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to churn based on customer demographics, services, contract details, and billing information.

## Objective

The objective of this project is to identify customers who are likely to leave the company and help businesses take proactive customer retention measures.

## Dataset

The project uses the Telco Customer Churn dataset, which contains customer information such as:

- Customer demographics
- Tenure
- Internet and phone services
- Contract type
- Payment method
- Monthly charges
- Total charges
- Churn status

## Exploratory Data Analysis

Performed exploratory data analysis to understand:

- Customer churn distribution
- Missing values
- Customer tenure
- Monthly and total charges
- Service usage patterns
- Contract and payment methods
- Factors associated with customer churn

## Data Preprocessing

- Handled missing values
- Converted data types
- Created tenure groups
- Converted categorical variables into dummy variables
- Converted the target variable into binary format
- Used SMOTEENN to handle class imbalance
- Applied StandardScaler for Logistic Regression

## Models Used

The following classification models were evaluated:

- Decision Tree
- Decision Tree with SMOTEENN
- Random Forest with SMOTEENN
- Logistic Regression with SMOTEENN
- SVM with SMOTEENN

## Final Model

Logistic Regression with SMOTEENN was selected as the final model based on its overall performance.

Accuracy: 94.06%

The model achieved balanced precision, recall, and F1-scores for both churn and non-churn customers.

## Deployment

The trained model is deployed using Streamlit. The application allows users to enter customer details and receive a prediction of whether the customer is likely to churn.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Streamlit
- Pickle
- Jupyter Notebook
- Git
- GitHub

## Project Structure

customer-churn-prediction/
│
├── data/
│   └── Telco Customer Churn Dataset
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   └── Telco Churn Analysis - EDA.ipynb
│
├── app.py
├── requirements.txt
└── README.md

## Live Demo

Streamlit App: https://customer-churn-prediction-9sfavuoi2mnpzigh5oktzk.streamlit.app/

## Author

Swathi Ponnaganti