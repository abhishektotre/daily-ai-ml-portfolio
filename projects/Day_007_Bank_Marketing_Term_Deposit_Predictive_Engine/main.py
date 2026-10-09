import os
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
