# Day 9: Real Estate Valuation with Stacking Regression Ensemble

![Domain](https://img.shields.io/badge/Domain-Machine%20Learning-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Accurate property appraisal requires capturing non-linear interactions, regional trends, and structural nuances. This project implements:
1. Multi-feature real estate property dataset (sqft, bedrooms, bathrooms, school rating, distance to city center, property age).
2. Base learners: Ridge Regression, Random Forest Regressor, and Gradient Boosting Regressor.
3. Meta-learner: StackingRegressor with cross-validated out-of-fold predictions.
4. Comprehensive regression error diagnostics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ determination coefficient.
5. Actual vs Predicted valuation scatter plot with regression fit line.

## 🛠️ Project Structure
```text
Day_009_Real_Estate_Valuation_Stacking_Ensemble/
├── data/
│   └── properties.csv
├── results/
│   ├── actual_vs_predicted.png
│   ├── residuals_plot.png
│   └── regression_metrics.json
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   └── stacking_engine.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_009_Real_Estate_Valuation_Stacking_Ensemble
pip install -r requirements.txt
python main.py
```
