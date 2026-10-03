# Day 8: Epidemiological Time-Series Trends & Public Health KPI Dashboard

![Domain](https://img.shields.io/badge/Domain-Data%20Analytics-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Epidemiological outbreak surveillance requires continuous time-series smoothing, peak wave identification, and hospital capacity monitoring. Inspired by epidemiological dashboards, this project implements:
1. Multi-wave daily outbreak telemetry generation across regional health authorities.
2. 7-day and 14-day centered/trailing moving average smoothing to eliminate weekend reporting artifacts.
3. Reproduction rate proxy ($R_t$) calculation using rolling incidence growth factors.
4. Active cases, recovery percentages, and healthcare surge index analytics.
5. Executive public health surveillance chart and KPI summary export.

## 🛠️ Project Structure
```text
Day_008_Epidemiological_Time_Series_Analytics_Dashboard/
├── data/
│   └── regional_epidemic_trends.csv
├── results/
│   ├── incidence_moving_averages.png
│   ├── reproduction_rate_trend.png
│   └── health_kpi_scorecard.json
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   └── timeseries_analyzer.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_008_Epidemiological_Time_Series_Analytics_Dashboard
pip install -r requirements.txt
python main.py
```
