import json
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
