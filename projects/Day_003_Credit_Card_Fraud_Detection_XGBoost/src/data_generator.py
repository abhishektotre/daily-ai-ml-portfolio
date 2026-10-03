import numpy as np
import pandas as pd
import os

def create_transactions(n_samples=5000, fraud_ratio=0.015, output_path="data/transactions.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # Legitimate features
    legit_amount = np.random.exponential(scale=65, size=n_legit).clip(2, 800)
    legit_dist = np.random.normal(loc=12, scale=8, size=n_legit).clip(0, 80)
    legit_vel = np.random.poisson(lam=1.5, size=n_legit)
    legit_foreign = np.random.choice([0, 1], size=n_legit, p=[0.95, 0.05])
    
    # Fraudulent features
    fraud_amount = np.random.exponential(scale=380, size=n_fraud).clip(50, 2500)
    fraud_dist = np.random.normal(loc=85, scale=40, size=n_fraud).clip(10, 400)
    fraud_vel = np.random.poisson(lam=4.8, size=n_fraud)
    fraud_foreign = np.random.choice([0, 1], size=n_fraud, p=[0.45, 0.55])
    
    amounts = np.concatenate([legit_amount, fraud_amount])
    distances = np.concatenate([legit_dist, fraud_dist])
    velocities = np.concatenate([legit_vel, fraud_vel])
    foreign = np.concatenate([legit_foreign, fraud_foreign])
    labels = np.array([0] * n_legit + [1] * n_fraud)
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    
    df = pd.DataFrame({
        "transaction_amount": amounts[idx].round(2),
        "distance_from_home_km": distances[idx].round(1),
        "velocity_transactions_1h": velocities[idx],
        "is_foreign_ip": foreign[idx],
        "is_fraud": labels[idx]
    })
    
    df.to_csv(output_path, index=False)
    return df
