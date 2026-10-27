import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_pipeline import generate_data
from src.engine import run_pipeline

def main():
    print("=" * 65)
    print(" 🚀 Running Day 25 Pipeline: Supply Chain Lead-Time & Disruption Predictor (Advanced Iteration 2)")
    print("=" * 65)
    
    print("[1/3] Generating domain-specific dataset...")
    df = generate_data()
    print(f"      Created {len(df)} observation records.")
    
    print("[2/3] Executing analytical modeling engine...")
    metrics = run_pipeline(df)
    
    print("[3/3] Execution Successful! Performance Metrics:")
    for k, v in metrics.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
