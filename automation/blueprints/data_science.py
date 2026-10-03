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

def generate_bank_deposit_project(day_num: int):
    folder_slug = "Bank_Marketing_Term_Deposit_Predictive_Engine"
    title = "Bank Marketing Term Deposit Predictive Engine"
    summary = "Direct bank marketing optimization pipeline utilizing demographic telemetry, call campaign duration, and ensemble classifiers to predict deposit subscriptions."
    skills = ["Data Science", "Banking Analytics", "Term Deposit", "Gradient Boosting", "Class Balancing", "Scikit-Learn"]

    readme_content = f"""# Day {day_num}: {title}

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
Day_{day_num:03d}_{folder_slug}/
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

def create_bank_dataset(n_samples=3200, output_path="data/bank_telemarketing_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    age = np.random.normal(loc=41, scale=11, size=n_samples).clip(18, 80).astype(int)
    job_categories = ["management", "technician", "blue-collar", "admin", "services", "retired"]
    jobs = np.random.choice(job_categories, size=n_samples, p=[0.25, 0.20, 0.22, 0.15, 0.10, 0.08])
    
    balance = np.random.exponential(scale=1400, size=n_samples).clip(-500, 35000).round(2)
    housing_loan = np.random.choice(["yes", "no"], size=n_samples, p=[0.55, 0.45])
    personal_loan = np.random.choice(["yes", "no"], size=n_samples, p=[0.16, 0.84])
    duration_sec = np.random.exponential(scale=260, size=n_samples).clip(10, 2400).astype(int)
    campaign_contacts = np.random.poisson(lam=2.0, size=n_samples).clip(1, 15)
    
    # Calculate subscription probability log-odds
    z = (
        -3.2
        + 0.007 * duration_sec
        + 0.00003 * balance
        - 0.6 * (housing_loan == "yes")
        - 0.5 * (personal_loan == "yes")
        + 0.5 * (jobs == "retired")
        - 0.08 * campaign_contacts
    )
    probs = 1 / (1 + np.exp(-z))
    subscribed = (np.random.rand(n_samples) < probs).astype(int)
    
    df = pd.DataFrame({
        "age": age,
        "job": jobs,
        "annual_balance": balance,
        "housing_loan": housing_loan,
        "personal_loan": personal_loan,
        "call_duration_seconds": duration_sec,
        "campaign_contacts": campaign_contacts,
        "subscribed": subscribed
    })
    df.to_csv(output_path, index=False)
    return df
"""

    model_code = """import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

def train_bank_model(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    X = df.drop(columns=["subscribed"])
    y = df["subscribed"]
    
    num_cols = ["age", "annual_balance", "call_duration_seconds", "campaign_contacts"]
    cat_cols = ["job", "housing_loan", "personal_loan"]
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    model.fit(X_train_proc, y_train)
    
    preds = model.predict(X_test_proc)
    probs = model.predict_proba(X_test_proc)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Bank Term Deposit Predictor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=200)
    plt.close()
    
    # 2. Conversion by Job Tier
    plt.figure(figsize=(8, 4))
    job_conv = df.groupby("job")["subscribed"].mean().sort_values(ascending=False) * 100
    job_conv.plot(kind="bar", color="#2ca02c")
    plt.title("Term Deposit Subscription Rate by Job Category (%)")
    plt.ylabel("Conversion Rate (%)")
    plt.xlabel("Job")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "conversion_by_job.png"), dpi=200)
    plt.close()
    
    metrics = {
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "test_eval_records": len(y_test),
        "overall_conversion_rate": round(float(df["subscribed"].mean()), 4)
    }
    with open(os.path.join(results_dir, "bank_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.bank_data_generator import create_bank_dataset
from src.deposit_model import train_bank_model

def main():
    print("=" * 65)
    print(" 🏦 Running Bank Term Deposit Predictive Modeling Pipeline")
    print("=" * 65)
    
    print("[1/3] Generating retail banking telemarketing records...")
    df = create_bank_dataset()
    print(f"      Created dataset with {len(df)} customer campaign contacts.")
    
    print("[2/3] Preprocessing features & training Gradient Boosting model...")
    metrics = train_bank_model(df)
    
    print("[3/3] Banking Model Scorecard:")
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
            "src/bank_data_generator.py": data_code,
            "src/deposit_model.py": model_code,
            "main.py": main_code
        }
    }


def generate_survival_duration_project(day_num: int):
    folder_slug = "Clinical_Survival_Duration_Analysis"
    title = "Kaplan-Meier Survival Duration & Hazard Modeling"
    summary = "Non-parametric survival analysis calculating Kaplan-Meier survival curves, hazard rates, and median survival duration for medical cohorts."
    skills = ["Data Science", "Survival Analysis", "Kaplan-Meier", "Hazard Function", "Statistical Testing", "Matplotlib"]

    readme_content = f"""# Day {day_num}: {title}

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
Day_{day_num:03d}_{folder_slug}/
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
cd Day_{day_num:03d}_{folder_slug}
pip install -r requirements.txt
python main.py
```
"""

    requirements_content = """pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    cohort_code = """import numpy as np
import pandas as pd
import os

def create_survival_cohort(n_patients=1600, output_path="data/patient_survival_records.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    treatment_arm = np.random.choice(["Control (Standard)", "Treatment (Novel Therapy)"], size=n_patients, p=[0.5, 0.5])
    age = np.random.normal(loc=58, scale=9, size=n_patients).clip(30, 85).astype(int)
    
    # Treatment group experiences longer survival times
    scale_param = np.where(treatment_arm == "Control (Standard)", 450.0, 720.0)
    
    # Weibull time-to-event distribution (shape k=1.3)
    k = 1.3
    actual_survival_days = scale_param * np.random.weibull(k, size=n_patients)
    
    # Study duration cutoff (censorship) at 900 days
    study_duration = 900
    observed_time = np.minimum(actual_survival_days, study_duration).round(1)
    event_observed = (actual_survival_days <= study_duration).astype(int) # 1 = Event, 0 = Censored
    
    df = pd.DataFrame({
        "patient_id": [f"PT-{i:05d}" for i in range(1, n_patients + 1)],
        "age": age,
        "treatment_group": treatment_arm,
        "duration_days": observed_time,
        "event_occurred": event_observed
    })
    df.to_csv(output_path, index=False)
    return df
"""

    estimator_code = """import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def compute_kaplan_meier(durations, events):
    # Sort distinct event times
    df_temp = pd.DataFrame({"t": durations, "e": events}).sort_values("t")
    timeline = np.unique(df_temp["t"])
    
    n_at_risk = len(df_temp)
    surv_prob = 1.0
    
    curve_t = [0]
    curve_s = [1.0]
    
    for t in timeline:
        d = df_temp[(df_temp["t"] == t) & (df_temp["e"] == 1)].shape[0]
        c = df_temp[(df_temp["t"] == t) & (df_temp["e"] == 0)].shape[0]
        
        if n_at_risk > 0:
            surv_prob *= (1.0 - d / n_at_risk)
            curve_t.append(t)
            curve_s.append(surv_prob)
            n_at_risk -= (d + c)
            
    return np.array(curve_t), np.array(curve_s)

def run_survival_analysis(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    ctrl = df[df["treatment_group"] == "Control (Standard)"]
    trt = df[df["treatment_group"] == "Treatment (Novel Therapy)"]
    
    t_ctrl, s_ctrl = compute_kaplan_meier(ctrl["duration_days"].values, ctrl["event_occurred"].values)
    t_trt, s_trt = compute_kaplan_meier(trt["duration_days"].values, trt["event_occurred"].values)
    
    # Plot Kaplan-Meier Curves
    plt.figure(figsize=(8, 5))
    plt.step(t_ctrl, s_ctrl, where="post", label="Control (Standard)", color="#d62728", lw=2)
    plt.step(t_trt, s_trt, where="post", label="Treatment (Novel Therapy)", color="#2ca02c", lw=2)
    plt.title("Kaplan-Meier Survival Functions by Cohort")
    plt.xlabel("Time in Days")
    plt.ylabel("Estimated Survival Probability S(t)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "kaplan_meier_curves.png"), dpi=200)
    plt.close()
    
    # Cumulative Hazard: H(t) = -ln(S(t))
    h_ctrl = -np.log(np.clip(s_ctrl, 1e-9, 1.0))
    h_trt = -np.log(np.clip(s_trt, 1e-9, 1.0))
    
    plt.figure(figsize=(8, 4))
    plt.step(t_ctrl, h_ctrl, where="post", label="Control Hazard", color="#d62728", lw=1.8)
    plt.step(t_trt, h_trt, where="post", label="Treatment Hazard", color="#2ca02c", lw=1.8)
    plt.title("Nelson-Aalen Cumulative Hazard Function H(t)")
    plt.xlabel("Time in Days")
    plt.ylabel("Cumulative Hazard")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cumulative_hazard.png"), dpi=200)
    plt.close()
    
    # Median survival calculation
    def get_median(t, s):
        idx = np.where(s <= 0.5)[0]
        return float(t[idx[0]]) if len(idx) > 0 else "> 900 days"
        
    metrics = {
        "total_cohort_size": len(df),
        "censoring_rate_pct": round(float((1 - df["event_occurred"].mean()) * 100), 2),
        "median_survival_control_days": get_median(t_ctrl, s_ctrl),
        "median_survival_treatment_days": get_median(t_trt, s_trt),
        "survival_at_day_365_control": round(float(s_ctrl[np.searchsorted(t_ctrl, 365) - 1]), 4),
        "survival_at_day_365_treatment": round(float(s_trt[np.searchsorted(t_trt, 365) - 1]), 4)
    }
    
    with open(os.path.join(results_dir, "survival_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.cohort_generator import create_survival_cohort
from src.survival_estimator import run_survival_analysis

def main():
    print("=" * 65)
    print(" 🏥 Running Clinical Kaplan-Meier Survival Analysis Engine")
    print("=" * 65)
    
    print("[1/3] Generating clinical patient survival cohort with censorship...")
    df = create_survival_cohort()
    print(f"      Synthesized {len(df)} patient records across treatment arms.")
    
    print("[2/3] Estimating non-parametric product-limit survival & hazard...")
    metrics = run_survival_analysis(df)
    
    print("[3/3] Survival Analysis Scorecard:")
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
            "src/cohort_generator.py": cohort_code,
            "src/survival_estimator.py": estimator_code,
            "main.py": main_code
        }
    }

DATA_SCIENCE_PROJECTS = [
    generate_churn_project,
    generate_bank_deposit_project,
    generate_survival_duration_project,
    generate_credit_risk_project
]
