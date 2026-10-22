# Day 20: E-Commerce Customer RFM Segmentation & Value Matrix

![Domain](https://img.shields.io/badge/Domain-Data%20Analytics-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Customer segmentation is fundamental to targeted marketing and profitability. This analytics engine:
1. Calculates granular Recency (days since last purchase), Frequency (order count), and Monetary (total spend) metrics.
2. Assigns quintile-based RFM scores (1-5 scale) creating composite customer tiers ('Champions', 'Loyalists', 'At Risk', 'Lost').
3. Applies K-Means clustering with automated standard scaling to discover natural behavioral segments.
4. Generates marketing strategy recommendations per customer segment.

## 🛠️ Project Structure
```text
Day_020_Customer_RFM_Segmentation_Matrix/
├── data/
│   └── ecommerce_orders.csv
├── results/
│   ├── rfm_segments_scatter.png
│   ├── segment_distribution.png
│   └── segment_summary.json
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   └── rfm_engine.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_020_Customer_RFM_Segmentation_Matrix
pip install -r requirements.txt
python main.py
```
