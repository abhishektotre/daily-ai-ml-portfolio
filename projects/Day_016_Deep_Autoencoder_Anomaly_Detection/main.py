import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_sensor_data
from src.autoencoder import train_autoencoder

def main():
    print("=" * 65)
    print(" 📉 Running Deep Autoencoder Unsupervised Anomaly Detection")
    print("=" * 65)
    
    print("[1/3] Generating multi-channel telemetry sensor streams...")
    df = create_sensor_data()
    print(f"      Synthesized {len(df)} samples with 8 correlated channels.")
    
    print("[2/3] Fitting Bottleneck Autoencoder [8 -> 4 -> 2 -> 4 -> 8]...")
    metrics = train_autoencoder(df)
    
    print("[3/3] Detection Performance:")
    for k, v in metrics.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
