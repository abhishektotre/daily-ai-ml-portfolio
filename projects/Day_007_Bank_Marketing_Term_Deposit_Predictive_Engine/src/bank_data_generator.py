import numpy as np
import pandas as pd
import os

def create_bank_dataset(n_samples=3200, output_path="data/bank_telemarketing_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    age = np.random.normal(loc=41, scale=11, size=n_samples).clip(18, 80).astype(int)
    job_categories = ["management", "technician", "blue-collar", "admin", "services", "retired"]
    jobs = np.random.choice(job_categories, size=n_samples, p=[0.25, 0.20, 0.22, 0.15, 0.10, 0.08])
    
    balance = np.random.exponential(scale=1400, size=n_samples).clip(-500, 35000).round(2)
    housing_loan = np.random.choice(["yes", "no"], size=n_samples, p=[0.55, 0.45])
    personal_loan = np.random.choice(["yes", "no"], size=n_samples, p=[0.16, 0.84])
    duration_sec = np.random.exponential(scale=260, size=n_samples).clip(10, 2400).astype(int)
    campaign_contacts = np.random.poisson(lam=2.0, size=n_samples).clip(1, 15)
    
    # Calculate subscription probability log-odds
    z = (
        -3.2
        + 0.007 * duration_sec
        + 0.00003 * balance
        - 0.6 * (housing_loan == "yes")
        - 0.5 * (personal_loan == "yes")
        + 0.5 * (jobs == "retired")
        - 0.08 * campaign_contacts
    )
    probs = 1 / (1 + np.exp(-z))
    subscribed = (np.random.rand(n_samples) < probs).astype(int)
    
    df = pd.DataFrame({
        "age": age,
        "job": jobs,
        "annual_balance": balance,
        "housing_loan": housing_loan,
        "personal_loan": personal_loan,
        "call_duration_seconds": duration_sec,
        "campaign_contacts": campaign_contacts,
        "subscribed": subscribed
    })
    df.to_csv(output_path, index=False)
    return df
