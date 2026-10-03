# Day 1: Customer Churn Prediction & Retention Analytics

![Domain](https://img.shields.io/badge/Domain-Data%20Science-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Customer churn poses a major threat to recurring revenue models. This project builds a complete, production-ready machine learning and survival analytics pipeline that:
1. Ingests and cleans multi-attribute customer telemetry data.
2. Performs comprehensive Exploratory Data Analysis (EDA) and identifies top attrition drivers.
3. Implements advanced feature engineering (tenure bins, charge-to-tenure ratio, contract risk scoring).
4. Trains and tunes predictive classification models (Logistic Regression & Random Forest).
5. Generates high-risk cohort profiles and action-oriented retention strategies.

## 🛠️ Project Structure
```text
Day_001_Customer_Churn_Prediction/
├── data/
│   └── customer_churn_data.csv    # Generated synthetic dataset
├── results/
│   ├── churn_distribution.png     # EDA target distribution
│   ├── feature_importance.png     # Top churn predictors
│   └── metrics.json               # Model performance scorecard
├── src/
│   ├── __init__.py
│   ├── data_generator.py          # Realistic telemetry synthesizer
│   ├── feature_engineering.py     # Data transformation and pipeline
│   └── model_trainer.py           # Training, tuning & evaluation
├── requirements.txt               # Project dependencies
├── main.py                        # Pipeline entrypoint
└── README.md
```

## 📊 Dataset Schema
- `customer_id`: Unique customer identifier
- `tenure_months`: Months the customer has been with the provider
- `monthly_charges`: Monthly billing amount ($)
- `total_charges`: Accumulated lifetime billing ($)
- `contract_type`: Month-to-month, One-year, or Two-year
- `tech_support`: Whether technical support is enabled (Yes/No)
- `online_security`: Subscribed to security suite (Yes/No)
- `churn`: Target indicator (1 = Churned, 0 = Retained)

## 🚀 How to Run
```bash
# 1. Navigate to project directory
cd Day_001_Customer_Churn_Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute the full end-to-end pipeline
python main.py
```

## 📈 Key Insights & Results
- Customers on **Month-to-Month contracts** with **high monthly charges** and **no technical support** exhibit an attrition probability 3.4x higher than annual subscribers.
- Feature importance analysis indicates `tenure_months` and `monthly_charges` are the highest ranking predictors.
