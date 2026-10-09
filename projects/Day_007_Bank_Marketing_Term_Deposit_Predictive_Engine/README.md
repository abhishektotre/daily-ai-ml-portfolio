# Day 7: Bank Marketing Term Deposit Predictive Engine

![Domain](https://img.shields.io/badge/Domain-Data%20Science-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Retail banks invest substantial capital in outbound telemarketing campaigns to secure term deposit subscriptions. Inspired by banking ML workflows, this project implements:
1. Multi-factor consumer banking telemetry synthesis (customer age, job tier, account balance, housing loan, campaign contact duration).
2. Advanced feature engineering capturing balance-to-age ratios and previous campaign engagement momentum.
3. Cost-sensitive gradient boosted ensemble classification addressing term deposit class imbalance.
4. Conversion lift analysis across demographic customer segments.
5. Actionable lead qualification scorecards for banking relationship managers.

## 🛠️ Project Structure
```text
Day_007_Bank_Marketing_Term_Deposit_Predictive_Engine/
├── data/
│   └── bank_telemarketing_data.csv
├── results/
│   ├── conversion_by_job.png
│   ├── roc_curve.png
│   └── bank_model_metrics.json
├── src/
│   ├── __init__.py
│   ├── bank_data_generator.py
│   └── deposit_model.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_007_Bank_Marketing_Term_Deposit_Predictive_Engine
pip install -r requirements.txt
python main.py
```
