# Day 14: E-Commerce Customer Transaction & Basket Value Analytics

![Domain](https://img.shields.io/badge/Domain-Data%20Analytics-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Retail business strategy relies heavily on understanding customer purchasing frequency and basket profitability. Inspired by transaction analytics workflows, this project implements:
1. Multi-category customer transaction ledger generation (order IDs, customer IDs, product categories, quantities, revenues).
2. Customer order frequency distribution analysis and Average Order Value (AOV) calculations.
3. Pareto 80/20 revenue concentration analysis (Lorenz curve formulation).
4. Basket item count vs transaction margin elasticity.
5. Executive merchandising scorecard with high-value customer tier rankings.

## 🛠️ Project Structure
```text
Day_014_ECommerce_Customer_Transaction_Analytics/
├── data/
│   └── customer_transactions.csv
├── results/
│   ├── pareto_revenue_curve.png
│   ├── basket_size_distribution.png
│   └── transaction_kpis.json
├── src/
│   ├── __init__.py
│   ├── transaction_generator.py
│   └── basket_analyzer.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_014_ECommerce_Customer_Transaction_Analytics
pip install -r requirements.txt
python main.py
```
