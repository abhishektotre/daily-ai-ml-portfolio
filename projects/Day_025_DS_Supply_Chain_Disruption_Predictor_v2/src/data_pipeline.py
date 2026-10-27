import numpy as np
import pandas as pd
import os

def generate_data(n_samples=2000, output_path="data/dataset.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42 + 25)
    
    f1 = np.random.normal(loc=50, scale=15, size=n_samples)
    f2 = np.random.exponential(scale=10, size=n_samples)
    f3 = np.random.uniform(1, 100, size=n_samples)
    f4 = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Target correlation
    signal = 0.04 * f1 + 0.08 * f2 - 0.02 * f3 + 1.2 * f4 + np.random.normal(0, 0.5, n_samples)
    prob = 1 / (1 + np.exp(-signal))
    target = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        "feature_metric_a": f1.round(2),
        "feature_metric_b": f2.round(2),
        "feature_metric_c": f3.round(2),
        "category_flag": f4,
        "target": target
    })
    df.to_csv(output_path, index=False)
    return df
