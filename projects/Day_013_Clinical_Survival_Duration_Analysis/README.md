# Day 13: Kaplan-Meier Survival Duration & Hazard Modeling

![Domain](https://img.shields.io/badge/Domain-Data%20Science-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Analyzing time-to-event data with censoring is fundamental in healthcare and reliability engineering. Inspired by survival duration estimation, this project implements:
1. Synthetic patient cohort telemetry (duration in days, censorship indicator, treatment group, baseline risk).
2. Pure mathematical Kaplan-Meier product-limit estimator from first principles.
3. Cumulative hazard rate calculation using the Nelson-Aalen estimator.
4. Stratified survival curve comparison between standard therapy and experimental treatment.
5. Median survival time and log-rank statistical disparity metric.

## 🛠️ Project Structure
```text
Day_013_Clinical_Survival_Duration_Analysis/
├── data/
│   └── patient_survival_records.csv
├── results/
│   ├── kaplan_meier_curves.png
│   ├── cumulative_hazard.png
│   └── survival_metrics.json
├── src/
│   ├── __init__.py
│   ├── cohort_generator.py
│   └── survival_estimator.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_013_Clinical_Survival_Duration_Analysis
pip install -r requirements.txt
python main.py
```
