# Day 19: Credit Risk Assessment & Default Probability Engine

![Domain](https://img.shields.io/badge/Domain-Data%20Science-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Financial credit assessment requires robust, interpretable scoring engines. This project implements:
1. Simulated multi-feature consumer credit profiles (debt-to-income, credit inquiries, revolving utilization, missed payments).
2. Weight of Evidence (WoE) and Information Value (IV) inspired binning and scaling.
3. Probability of Default (PD) calibrated logistic regression model.
4. Production credit scorecard scaling (Base Score 600, Points to Double Odds = 20).
5. Statistical validation using Kolmogorov-Smirnov (KS) statistic and ROC-AUC curve.

## 🛠️ Project Structure
```text
Day_019_Credit_Risk_Assessment_Engine/
├── data/
│   └── credit_loan_data.csv       # Borrower historical profiles
├── results/
│   ├── credit_score_distribution.png
│   ├── roc_curve.png
│   └── metrics.json
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   └── credit_engine.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_019_Credit_Risk_Assessment_Engine
pip install -r requirements.txt
python main.py
```
