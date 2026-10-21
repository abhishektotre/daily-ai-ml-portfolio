import numpy as np
import pandas as pd
import os

def create_loan_dataset(n_samples=3000, random_state=42, output_path="data/credit_loan_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(random_state)
    
    annual_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples).clip(20000, 250000).round(-2)
    debt_to_income = np.random.beta(a=2, b=5, size=n_samples) * 0.65
    revolving_util = np.random.beta(a=3, b=3, size=n_samples)
    recent_inquiries = np.random.poisson(lam=1.2, size=n_samples).clip(0, 10)
    delinquencies = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.75, 0.15, 0.07, 0.03])
    loan_amount = (annual_income * np.random.uniform(0.1, 0.45, size=n_samples)).round(-2)
    
    z = (
        -2.5
        + 3.2 * debt_to_income
        + 2.8 * revolving_util
        + 0.3 * recent_inquiries
        + 0.9 * delinquencies
        - 0.000015 * annual_income
    )
    prob_default = 1 / (1 + np.exp(-z))
    defaulted = (np.random.rand(n_samples) < prob_default).astype(int)
    
    df = pd.DataFrame({
        "annual_income": annual_income,
        "debt_to_income": debt_to_income.round(4),
        "revolving_utilization": revolving_util.round(4),
        "inquiries_last_6m": recent_inquiries,
        "delinquencies_2yrs": delinquencies,
        "loan_amount": loan_amount,
        "default": defaulted
    })
    df.to_csv(output_path, index=False)
    return df
