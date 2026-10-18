import numpy as np
import pandas as pd
import os

def create_sensor_data(n_samples=3000, anomaly_ratio=0.03, output_path="data/sensor_streams.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    # 8 correlated sensor dimensions
    t = np.linspace(0, 50, n_samples)
    s1 = np.sin(t) + np.random.normal(0, 0.1, n_samples)
    s2 = np.cos(t * 0.5) + np.random.normal(0, 0.1, n_samples)
    s3 = 0.5 * s1 + 0.8 * s2 + np.random.normal(0, 0.05, n_samples)
    s4 = np.sin(t * 1.5) + np.random.normal(0, 0.15, n_samples)
    s5 = -0.7 * s2 + np.random.normal(0, 0.1, n_samples)
    s6 = np.tanh(s1) + np.random.normal(0, 0.05, n_samples)
    s7 = s4 * 0.6 + s5 * 0.4 + np.random.normal(0, 0.1, n_samples)
    s8 = np.exp(-0.01 * t) * np.sin(t) + np.random.normal(0, 0.1, n_samples)
    
    data = np.column_stack([s1, s2, s3, s4, s5, s6, s7, s8])
    labels = np.zeros(n_samples, dtype=int)
    
    # Inject anomaly bursts
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
    for idx in anomaly_indices:
        data[idx] += np.random.uniform(2.5, 5.0, size=8) * np.random.choice([-1, 1], size=8)
        labels[idx] = 1
        
    cols = [f"sensor_{i}" for i in range(1, 9)]
    df = pd.DataFrame(data, columns=cols)
    df["is_anomaly"] = labels
    df.to_csv(output_path, index=False)
    return df
