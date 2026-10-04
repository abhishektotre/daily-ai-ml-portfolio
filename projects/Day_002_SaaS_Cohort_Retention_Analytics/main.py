import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_event_logs
from src.cohort_analyzer import analyze_cohorts

def main():
    print("=" * 65)
    print(" 📊 Running SaaS Cohort Retention & Churn Analytics Pipeline")
    print("=" * 65)
    
    print("[1/3] Generating subscription transaction events...")
    df = create_event_logs()
    print(f"      Created {len(df)} billing transactions across users.")
    
    print("[2/3] Computing cohort index, retention matrix & visualization heatmaps...")
    metrics = analyze_cohorts(df)
    
    print("[3/3] Analytics Run Complete! Executive Summary:")
    for k, v in metrics.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
