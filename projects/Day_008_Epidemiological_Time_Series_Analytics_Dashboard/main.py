import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_epidemic_data
from src.timeseries_analyzer import analyze_epidemic_timeseries

def main():
    print("=" * 65)
    print(" 🏥 Running Public Health Epidemiological Time-Series Engine")
    print("=" * 65)
    
    print("[1/3] Generating regional daily outbreak transmission logs...")
    df = create_epidemic_data()
    print(f"      Synthesized {len(df)} days of continuous surveillance telemetry.")
    
    print("[2/3] Calculating 7-day smoothing & Rt reproduction rate curves...")
    kpis = analyze_epidemic_timeseries(df)
    
    print("[3/3] Public Health Executive KPI Scorecard:")
    for k, v in kpis.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
