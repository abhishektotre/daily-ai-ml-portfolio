import json
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
