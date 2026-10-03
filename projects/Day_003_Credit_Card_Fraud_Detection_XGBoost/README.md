# Day 3: High-Precision Financial Fraud Detection with XGBoost

![Domain](https://img.shields.io/badge/Domain-Machine%20Learning-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Detecting financial fraud requires catching rare fraudulent transactions (often < 1% of transactions) while minimizing false positives that frustrate legitimate cardholders. This project implements:
1. Highly imbalanced transactional dataset generator (0.8% fraud prevalence).
2. Advanced feature scaling with RobustScaler (resistant to financial outliers).
3. Extreme Gradient Boosting (XGBoost) classifier parameterized with `scale_pos_weight` to address imbalance.
4. Precision-Recall AUC (PR-AUC) and cost-benefit threshold optimization.
5. Confusion matrix and feature importance visualization.

## 🛠️ Project Structure
```text
Day_003_Credit_Card_Fraud_Detection_XGBoost/
├── data/
│   └── transactions.csv
├── results/
│   ├── confusion_matrix.png
│   ├── pr_curve.png
│   └── evaluation_scorecard.json
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   └── model_pipeline.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_003_Credit_Card_Fraud_Detection_XGBoost
pip install -r requirements.txt
python main.py
```
