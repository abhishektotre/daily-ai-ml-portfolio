import os
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
