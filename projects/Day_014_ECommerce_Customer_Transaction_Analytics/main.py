import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.transaction_generator import create_transaction_ledger
from src.basket_analyzer import analyze_transactions

def main():
    print("=" * 65)
    print(" 🛒 Running E-Commerce Transaction & Basket Value Analytics")
    print("=" * 65)
    
    print("[1/3] Generating multi-item customer transaction ledger...")
    df = create_transaction_ledger()
    print(f"      Created ledger with {len(df)} retail orders.")
    
    print("[2/3] Computing Pareto Lorenz curve & basket size distributions...")
    kpis = analyze_transactions(df)
    
    print("[3/3] Transaction Analytics Summary:")
    for k, v in kpis.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
