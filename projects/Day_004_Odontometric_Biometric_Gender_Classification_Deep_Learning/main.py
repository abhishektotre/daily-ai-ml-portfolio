import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_odontometric_dataset
from src.deep_classifier import train_biometric_model

def main():
    print("=" * 65)
    print(" 🦷 Running Deep Odontometric Biometric Classification")
    print("=" * 65)
    
    print("[1/3] Synthesizing clinical odontometric measurement cohort...")
    df = create_odontometric_dataset()
    print(f"      Generated {len(df)} patient dental records.")
    
    print("[2/3] Training Deep MLP Network [64 -> 32 -> 16] with Adam...")
    metrics = train_biometric_model(df)
    
    print("[3/3] Biometric Classification Results:")
    for k, v in metrics.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
