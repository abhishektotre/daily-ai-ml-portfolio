import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.cohort_generator import create_survival_cohort
from src.survival_estimator import run_survival_analysis

def main():
    print("=" * 65)
    print(" 🏥 Running Clinical Kaplan-Meier Survival Analysis Engine")
    print("=" * 65)
    
    print("[1/3] Generating clinical patient survival cohort with censorship...")
    df = create_survival_cohort()
    print(f"      Synthesized {len(df)} patient records across treatment arms.")
    
    print("[2/3] Estimating non-parametric product-limit survival & hazard...")
    metrics = run_survival_analysis(df)
    
    print("[3/3] Survival Analysis Scorecard:")
    for k, v in metrics.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
