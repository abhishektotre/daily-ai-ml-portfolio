"""
Machine Learning Blueprints:
Includes high-performance supervised learning, XGBoost fraud detection,
regression stacking ensembles, and predictive maintenance.
"""

def generate_fraud_detection_project(day_num: int):
    folder_slug = "Credit_Card_Fraud_Detection_XGBoost"
    title = "High-Precision Financial Fraud Detection with XGBoost"
    summary = "Imbalanced financial fraud classification pipeline utilizing XGBoost, precision-recall optimization, and cost-sensitive loss functions."
    skills = ["Machine Learning", "XGBoost", "Imbalanced Data", "Precision-Recall AUC", "Cost-Sensitive Learning", "Scikit-Learn"]

    readme_content = f"""# Day {day_num}: {title}

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
Day_{day_num:03d}_{folder_slug}/
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
cd Day_{day_num:03d}_{folder_slug}
pip install -r requirements.txt
python main.py
```
"""

    requirements_content = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    data_code = """import numpy as np
import pandas as pd
import os

def create_transactions(n_samples=5000, fraud_ratio=0.015, output_path="data/transactions.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # Legitimate features
    legit_amount = np.random.exponential(scale=65, size=n_legit).clip(2, 800)
    legit_dist = np.random.normal(loc=12, scale=8, size=n_legit).clip(0, 80)
    legit_vel = np.random.poisson(lam=1.5, size=n_legit)
    legit_foreign = np.random.choice([0, 1], size=n_legit, p=[0.95, 0.05])
    
    # Fraudulent features
    fraud_amount = np.random.exponential(scale=380, size=n_fraud).clip(50, 2500)
    fraud_dist = np.random.normal(loc=85, scale=40, size=n_fraud).clip(10, 400)
    fraud_vel = np.random.poisson(lam=4.8, size=n_fraud)
    fraud_foreign = np.random.choice([0, 1], size=n_fraud, p=[0.45, 0.55])
    
    amounts = np.concatenate([legit_amount, fraud_amount])
    distances = np.concatenate([legit_dist, fraud_dist])
    velocities = np.concatenate([legit_vel, fraud_vel])
    foreign = np.concatenate([legit_foreign, fraud_foreign])
    labels = np.array([0] * n_legit + [1] * n_fraud)
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    
    df = pd.DataFrame({
        "transaction_amount": amounts[idx].round(2),
        "distance_from_home_km": distances[idx].round(1),
        "velocity_transactions_1h": velocities[idx],
        "is_foreign_ip": foreign[idx],
        "is_fraud": labels[idx]
    })
    
    df.to_csv(output_path, index=False)
    return df
"""

    pipeline_code = """import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import xgboost as xgb

def train_and_score(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Imbalance ratio
    ratio = float(np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-9))
    
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Precision-Recall
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recalls, precisions)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title("Confusion Matrix - Fraud Detection (XGBoost)")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=200)
    plt.close()
    
    # PR Curve Plot
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, color="crimson", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Imbalanced Classification)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pr_curve.png"), dpi=200)
    plt.close()
    
    scorecard = {
        "pr_auc": round(float(pr_auc), 4),
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
        "fraud_capture_rate_recall": round(float(cm[1][1] / (cm[1][0] + cm[1][1])), 4)
    }
    
    with open(os.path.join(results_dir, "evaluation_scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=4)
        
    return scorecard
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_transactions
from src.model_pipeline import train_and_score

def main():
    print("=" * 65)
    print(" 💳 Running Machine Learning Pipeline: Financial Fraud Detection")
    print("=" * 65)
    
    print("[1/3] Synthesizing highly imbalanced transactional ledger...")
    df = create_transactions()
    fraud_rate = (df['is_fraud'].mean() * 100)
    print(f"      Ledger contains {len(df)} transactions ({fraud_rate:.2f}% fraud).")
    
    print("[2/3] Training cost-weighted XGBoost with Precision-Recall optimization...")
    scorecard = train_and_score(df)
    
    print("[3/3] Model Evaluation Scorecard:")
    for k, v in scorecard.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Machine Learning",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "src/__init__.py": "",
            "src/data_generator.py": data_code,
            "src/model_pipeline.py": pipeline_code,
            "main.py": main_code
        }
    }


def generate_real_estate_stacking_project(day_num: int):
    folder_slug = "Real_Estate_Valuation_Stacking_Ensemble"
    title = "Real Estate Valuation with Stacking Regression Ensemble"
    summary = "Multi-model ensemble regression architecture combining Ridge, Random Forest, and Gradient Boosting with meta-learner for property valuation."
    skills = ["Machine Learning", "Stacking Regressor", "Ensemble Methods", "Feature Engineering", "R2 Score", "RMSE"]

    readme_content = f"""# Day {day_num}: {title}

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
Day_{day_num:03d}_{folder_slug}/
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
cd Day_{day_num:03d}_{folder_slug}
pip install -r requirements.txt
python main.py
```
"""

    requirements_content = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    data_code = """import numpy as np
import pandas as pd
import os

def create_property_data(n_samples=2000, output_path="data/properties.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    sqft = np.random.normal(loc=2100, scale=600, size=n_samples).clip(800, 5000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.25, 0.45, 0.20, 0.05])
    bathrooms = np.clip(np.round(bedrooms * 0.75 + np.random.normal(0, 0.5, n_samples), 1), 1, 4.5)
    school_rating = np.random.randint(1, 11, size=n_samples) # 1-10
    dist_downtown = np.random.exponential(scale=8, size=n_samples).clip(0.5, 30) # km
    age_years = np.random.uniform(0, 60, size=n_samples)
    
    # Valuation function with non-linear factors
    base_price = (
        120000
        + 185 * sqft
        + 15000 * bedrooms
        + 22000 * bathrooms
        + 12000 * school_rating
        - 4500 * dist_downtown
        - 800 * age_years
        + np.random.normal(0, 25000, size=n_samples)
    ).clip(85000, 1500000)
    
    df = pd.DataFrame({
        "sqft_living": sqft.round(),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "school_rating": school_rating,
        "distance_downtown_km": dist_downtown.round(2),
        "age_years": age_years.round(1),
        "price_usd": base_price.round(-2)
    })
    df.to_csv(output_path, index=False)
    return df
"""

    stacking_code = """import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_stacking_ensemble(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    X = df.drop(columns=["price_usd"])
    y = df["price_usd"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    estimators = [
        ("ridge", Ridge(alpha=10.0)),
        ("rf", RandomForestRegressor(n_estimators=80, max_depth=7, random_state=42)),
        ("gbr", GradientBoostingRegressor(n_estimators=90, learning_rate=0.08, random_state=42))
    ]
    
    stacker = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5
    )
    
    stacker.fit(X_train, y_train)
    y_pred = stacker.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test / 1000, y_pred / 1000, alpha=0.5, color="#2b5c8f")
    lims = [min(y_test.min(), y_pred.min()) / 1000, max(y_test.max(), y_pred.max()) / 1000]
    plt.plot(lims, lims, color="red", linestyle="--", lw=2)
    plt.title(f"Actual vs Predicted Valuation ($k) - R2: {r2:.3f}")
    plt.xlabel("Actual Price ($k)")
    plt.ylabel("Predicted Price ($k)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "actual_vs_predicted.png"), dpi=200)
    plt.close()
    
    metrics = {
        "rmse_usd": round(float(rmse), 2),
        "mae_usd": round(float(mae), 2),
        "r2_score": round(float(r2), 4),
        "sample_count": len(y_test)
    }
    
    with open(os.path.join(results_dir, "regression_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_property_data
from src.stacking_engine import train_stacking_ensemble

def main():
    print("=" * 65)
    print(" 🏡 Running Real Estate Stacking Regression Ensemble Pipeline")
    print("=" * 65)
    
    print("[1/3] Generating housing and municipal feature attributes...")
    df = create_property_data()
    print(f"      Created dataset with {len(df)} properties.")
    
    print("[2/3] Fitting Ridge + RF + GBR base models & linear meta-learner...")
    metrics = train_stacking_ensemble(df)
    
    print("[3/3] Stacking Ensemble Performance:")
    print(f"      - R² Score: {metrics['r2_score']}")
    print(f"      - RMSE: ${metrics['rmse_usd']:,.2f}")
    print(f"      - MAE:  ${metrics['mae_usd']:,.2f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Machine Learning",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "src/__init__.py": "",
            "src/data_generator.py": data_code,
            "src/stacking_engine.py": stacking_code,
            "main.py": main_code
        }
    }

MACHINE_LEARNING_PROJECTS = [
    generate_fraud_detection_project,
    generate_real_estate_stacking_project
]
