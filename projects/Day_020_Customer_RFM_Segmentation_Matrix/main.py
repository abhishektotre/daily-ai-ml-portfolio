import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_order_data
from src.rfm_engine import run_rfm_analysis

def main():
    print("=" * 65)
    print(" 🛍️ Running Customer RFM Segmentation & Value Matrix Pipeline")
    print("=" * 65)
    
    print("[1/3] Generating e-commerce multi-order transaction history...")
    df = create_order_data()
    print(f"      Generated {len(df)} transactions.")
    
    print("[2/3] Performing RFM calculation, quantile scoring & tier segmentation...")
    summary = run_rfm_analysis(df)
    
    print("[3/3] Segmentation Complete! Key Summary:")
    print(f"      - Total Customers Analyzed: {summary['total_customers']}")
    print(f"      - Average Customer Spend: ${summary['avg_customer_value']}")
    print("      - Segments Breakdown:")
    for seg, count in summary["segments"].items():
        print(f"        * {seg}: {count} customers")
    print("=" * 65)

if __name__ == "__main__":
    main()
