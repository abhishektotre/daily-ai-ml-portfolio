"""
Day 043: Customer Lifetime Value & Survival Duration Modeling (Advanced Iteration 4)
Main Pipeline Entrypoint.
"""

import os
import sys

from src.data_pipeline import generate_data
from src.engine import run_pipeline

def main():
    print("=" * 65)
    print(" 🚀 Running Day 43 Pipeline: Customer Lifetime Value & Survival Duration Modeling (Advanced Iteration 4)")
    print("=" * 65)
    
    print("[1/3] Generating synthetic observation dataset...")
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
