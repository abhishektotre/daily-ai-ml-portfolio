# Day 2: SaaS Monthly Cohort Retention & Churn Analytics

![Domain](https://img.shields.io/badge/Domain-Data%20Analytics-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Understanding user retention over time is the lifeblood of subscription & SaaS businesses. This project provides an enterprise-grade analytics engine that:
1. Simulates realistic recurring user event logs over a 12-month timeline.
2. Derives cohort acquisition months and relative cohort index (Month 0 to Month 11).
3. Constructs user retention percentage matrices.
4. Generates an automated Seaborn cohort retention heatmap for executive dashboards.
5. Calculates average customer survival half-life and monthly churn rates.

## 🛠️ Project Structure
```text
Day_002_SaaS_Cohort_Retention_Analytics/
├── data/
│   └── user_subscriptions.csv     # Event transaction logs
├── results/
│   ├── cohort_retention_heatmap.png
│   ├── monthly_active_users.png
│   └── cohort_metrics.json
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   └── cohort_analyzer.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_002_SaaS_Cohort_Retention_Analytics
pip install -r requirements.txt
python main.py
```
