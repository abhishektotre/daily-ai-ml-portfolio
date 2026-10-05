import json
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
