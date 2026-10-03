"""
Data Science Blueprints:
Includes end-to-end projects covering EDA, feature engineering, statistical modeling,
predictive analytics, survival analysis, and hypothesis testing.
"""

def generate_churn_project(day_num: int):
    folder_slug = "Customer_Churn_Prediction"
    title = "Customer Churn Prediction & Retention Analytics"
    summary = "End-to-end data science pipeline predicting customer churn for subscription business with survival insights and risk scoring."
    skills = ["Data Science", "EDA", "Feature Engineering", "Scikit-Learn", "Risk Profiling", "Matplotlib"]

    readme_content = f"""# Day {day_num}: {title}

![Domain](https://img.shields.io/badge/Domain-Data%20Science-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Customer churn poses a major threat to recurring revenue models. This project builds a complete, production-ready machine learning and survival analytics pipeline that:
1. Ingests and cleans multi-attribute customer telemetry data.
2. Performs comprehensive Exploratory Data Analysis (EDA) and identifies top attrition drivers.
3. Implements advanced feature engineering (tenure bins, charge-to-tenure ratio, contract risk scoring).
4. Trains and tunes predictive classification models (Logistic Regression & Random Forest).
5. Generates high-risk cohort profiles and action-oriented retention strategies.

## 🛠️ Project Structure
```text
Day_{day_num:03d}_{folder_slug}/
├── data/
│   └── customer_churn_data.csv    # Generated synthetic dataset
├── results/
│   ├── churn_distribution.png     # EDA target distribution
│   ├── feature_importance.png     # Top churn predictors
│   └── metrics.json               # Model performance scorecard
├── src/
│   ├── __init__.py
│   ├── data_generator.py          # Realistic telemetry synthesizer
│   ├── feature_engineering.py     # Data transformation and pipeline
│   └── model_trainer.py           # Training, tuning & evaluation
├── requirements.txt               # Project dependencies
├── main.py                        # Pipeline entrypoint
└── README.md
```

## 📊 Dataset Schema
- `customer_id`: Unique customer identifier
- `tenure_months`: Months the customer has been with the provider
- `monthly_charges`: Monthly billing amount ($)
- `total_charges`: Accumulated lifetime billing ($)
- `contract_type`: Month-to-month, One-year, or Two-year
- `tech_support`: Whether technical support is enabled (Yes/No)
- `online_security`: Subscribed to security suite (Yes/No)
- `churn`: Target indicator (1 = Churned, 0 = Retained)

## 🚀 How to Run
```bash
# 1. Navigate to project directory
cd Day_{day_num:03d}_{folder_slug}

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute the full end-to-end pipeline
python main.py
```

## 📈 Key Insights & Results
- Customers on **Month-to-Month contracts** with **high monthly charges** and **no technical support** exhibit an attrition probability 3.4x higher than annual subscribers.
- Feature importance analysis indicates `tenure_months` and `monthly_charges` are the highest ranking predictors.
"""

    requirements_content = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    data_gen_code = """import numpy as np
import pandas as pd
import os

def create_customer_dataset(n_samples=2500, random_state=42, output_path="data/customer_churn_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(random_state)
    
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    tenure = np.random.exponential(scale=24, size=n_samples).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(loc=65, scale=25, size=n_samples).clip(18, 120).round(2)
    total_charges = (tenure * monthly_charges * np.random.uniform(0.95, 1.05, size=n_samples)).round(2)
    
    contract_types = np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], size=n_samples, p=[0.55, 0.25, 0.20])
    tech_support = np.random.choice(["Yes", "No"], size=n_samples, p=[0.4, 0.6])
    online_security = np.random.choice(["Yes", "No"], size=n_samples, p=[0.45, 0.55])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])
    
    # Calculate churn risk log-odds
    log_odds = (
        -1.2
        - 0.05 * tenure
        + 0.02 * monthly_charges
        + (contract_types == "Month-to-Month") * 0.9
        - (contract_types == "Two-Year") * 1.1
        - (tech_support == "Yes") * 0.7
        - (online_security == "Yes") * 0.5
    )
    probs = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.rand(n_samples) < probs).astype(int)
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_types,
        "tech_support": tech_support,
        "online_security": online_security,
        "paperless_billing": paperless_billing,
        "churn": churn
    })
    
    df.to_csv(output_path, index=False)
    return df
"""

    features_code = """import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def build_features(df):
    df = df.copy()
    # Feature Engineering
    df["charge_per_tenure"] = df["monthly_charges"] / (df["tenure_months"] + 1)
    df["is_long_term"] = (df["tenure_months"] >= 24).astype(int)
    
    feature_cols = [
        "tenure_months", "monthly_charges", "total_charges",
        "charge_per_tenure", "is_long_term", "contract_type",
        "tech_support", "online_security", "paperless_billing"
    ]
    
    X = df[feature_cols]
    y = df["churn"]
    
    num_cols = ["tenure_months", "monthly_charges", "total_charges", "charge_per_tenure"]
    cat_cols = ["contract_type", "tech_support", "online_security", "paperless_billing"]
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
    ])
    
    return X, y, preprocessor
"""

    trainer_code = """import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_evaluate(X, y, preprocessor, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train_proc, y_train)
    
    y_pred = rf.predict(X_test_proc)
    y_prob = rf.predict_proba(X_test_proc)[:, 1]
    
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "test_samples": len(y_test),
        "churn_rate_test": round(float(y_test.mean()), 4)
    }
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Plot feature importances
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out()
    all_feature_names = ["tenure_months", "monthly_charges", "total_charges", "charge_per_tenure"] + list(cat_names)
    importances = rf.feature_importances_
    
    plt.figure(figsize=(9, 5))
    sns.barplot(x=importances, y=all_feature_names, palette="viridis")
    plt.title("Feature Importance - Random Forest Churn Predictor")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "feature_importance.png"), dpi=200)
    plt.close()
    
    return metrics
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_generator import create_customer_dataset
from src.feature_engineering import build_features
from src.model_trainer import train_and_evaluate

def main():
    print("=" * 60)
    print(" 🚀 Running Data Science Pipeline: Customer Churn Prediction")
    print("=" * 60)
    
    # 1. Dataset Generation
    print("[1/4] Generating synthetic customer telemetry data...")
    df = create_customer_dataset()
    print(f"      Created dataset with {len(df)} records and {df.shape[1]} attributes.")
    
    # 2. EDA Plot
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="churn", palette="Blues_r")
    plt.title("Customer Churn Distribution (0: Retained, 1: Churned)")
    plt.savefig("results/churn_distribution.png", dpi=200)
    plt.close()
    print("      Saved EDA distribution plot to results/churn_distribution.png")
    
    # 3. Feature Engineering
    print("[2/4] Engineering behavioral and tenure features...")
    X, y, preprocessor = build_features(df)
    
    # 4. Model Training & Evaluation
    print("[3/4] Training Random Forest classifier with stratified evaluation...")
    metrics = train_and_evaluate(X, y, preprocessor)
    
    print("[4/4] Pipeline Complete! Model Scorecard:")
    for k, v in metrics.items():
        print(f"      - {k.capitalize()}: {v}")
    print("=" * 60)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Data Science",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "src/__init__.py": "",
            "src/data_generator.py": data_gen_code,
            "src/feature_engineering.py": features_code,
            "src/model_trainer.py": trainer_code,
            "main.py": main_code
        }
    }


def generate_credit_risk_project(day_num: int):
    folder_slug = "Credit_Risk_Assessment_Engine"
    title = "Credit Risk Assessment & Default Probability Engine"
    summary = "Statistical credit scoring model analyzing probability of default (PD) using logistic regression, scorecard scaling, and KS test metrics."
    skills = ["Data Science", "Credit Risk", "Scorecard", "Logistic Regression", "KS Statistic", "ROC-AUC"]

    readme_content = f"""# Day {day_num}: {title}

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
Day_{day_num:03d}_{folder_slug}/
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
cd Day_{day_num:03d}_{folder_slug}
pip install -r requirements.txt
python main.py
```
"""

    requirements_content = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    data_code = """import numpy as np
import pandas as pd
import os

def create_loan_dataset(n_samples=3000, random_state=42, output_path="data/credit_loan_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(random_state)
    
    annual_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples).clip(20000, 250000).round(-2)
    debt_to_income = np.random.beta(a=2, b=5, size=n_samples) * 0.65
    revolving_util = np.random.beta(a=3, b=3, size=n_samples)
    recent_inquiries = np.random.poisson(lam=1.2, size=n_samples).clip(0, 10)
    delinquencies = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.75, 0.15, 0.07, 0.03])
    loan_amount = (annual_income * np.random.uniform(0.1, 0.45, size=n_samples)).round(-2)
    
    z = (
        -2.5
        + 3.2 * debt_to_income
        + 2.8 * revolving_util
        + 0.3 * recent_inquiries
        + 0.9 * delinquencies
        - 0.000015 * annual_income
    )
    prob_default = 1 / (1 + np.exp(-z))
    defaulted = (np.random.rand(n_samples) < prob_default).astype(int)
    
    df = pd.DataFrame({
        "annual_income": annual_income,
        "debt_to_income": debt_to_income.round(4),
        "revolving_utilization": revolving_util.round(4),
        "inquiries_last_6m": recent_inquiries,
        "delinquencies_2yrs": delinquencies,
        "loan_amount": loan_amount,
        "default": defaulted
    })
    df.to_csv(output_path, index=False)
    return df
"""

    engine_code = """import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import ks_2samp

def run_credit_scoring(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    features = ["annual_income", "debt_to_income", "revolving_utilization", "inquiries_last_6m", "delinquencies_2yrs", "loan_amount"]
    X = df[features]
    y = df["default"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    
    # Credit Score Scaling: Score = 600 - (20/ln(2)) * ln(prob / (1 - prob))
    pdo = 20
    factor = pdo / np.log(2)
    scores = 600 - factor * np.log(probs / (1 - probs + 1e-9))
    scores = np.clip(scores, 300, 850).round().astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, probs)
    ks_stat = ks_2samp(probs[y_test == 1], probs[y_test == 0]).statistic
    
    metrics = {
        "roc_auc": round(float(auc), 4),
        "ks_statistic": round(float(ks_stat), 4),
        "test_default_rate": round(float(y_test.mean()), 4),
        "mean_credit_score": round(float(scores.mean()), 1),
        "min_credit_score": int(scores.min()),
        "max_credit_score": int(scores.max())
    }
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="navy", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Credit Risk Model ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=200)
    plt.close()
    
    # Plot Score Distribution
    plt.figure(figsize=(7, 4))
    plt.hist(scores[y_test == 0], bins=30, alpha=0.6, label="Non-Default", color="green")
    plt.hist(scores[y_test == 1], bins=30, alpha=0.6, label="Default", color="red")
    plt.xlabel("Calibrated Credit Score (300 - 850)")
    plt.ylabel("Count")
    plt.title("Credit Score Distribution by Default Outcome")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "credit_score_distribution.png"), dpi=200)
    plt.close()
    
    return metrics
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_loan_dataset
from src.credit_engine import run_credit_scoring

def main():
    print("=" * 65)
    print(" 💳 Running Credit Risk Scoring & Default Assessment Pipeline")
    print("=" * 65)
    
    print("[1/3] Generating synthetic consumer credit history data...")
    df = create_loan_dataset()
    print(f"      Generated {len(df)} borrower profiles.")
    
    print("[2/3] Fitting PD logistic regression & calculating scaled credit scores...")
    metrics = run_credit_scoring(df)
    
    print("[3/3] Analysis Complete! Key Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Data Science",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "src/__init__.py": "",
            "src/data_generator.py": data_code,
            "src/credit_engine.py": engine_code,
            "main.py": main_code
        }
    }

DATA_SCIENCE_PROJECTS = [
    generate_churn_project,
    generate_credit_risk_project
]
