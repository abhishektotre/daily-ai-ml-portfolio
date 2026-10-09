import json
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
