import os
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
