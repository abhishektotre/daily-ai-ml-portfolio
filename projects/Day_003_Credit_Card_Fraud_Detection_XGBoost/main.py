import os
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
