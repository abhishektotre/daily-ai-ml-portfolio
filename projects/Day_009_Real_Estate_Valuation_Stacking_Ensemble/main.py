import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_property_data
from src.stacking_engine import train_stacking_ensemble

def main():
    print("=" * 65)
    print(" 🏡 Running Real Estate Stacking Regression Ensemble Pipeline")
    print("=" * 65)
    
    print("[1/3] Generating housing and municipal feature attributes...")
    df = create_property_data()
    print(f"      Created dataset with {len(df)} properties.")
    
    print("[2/3] Fitting Ridge + RF + GBR base models & linear meta-learner...")
    metrics = train_stacking_ensemble(df)
    
    print("[3/3] Stacking Ensemble Performance:")
    print(f"      - R² Score: {metrics['r2_score']}")
    print(f"      - RMSE: ${metrics['rmse_usd']:,.2f}")
    print(f"      - MAE:  ${metrics['mae_usd']:,.2f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
