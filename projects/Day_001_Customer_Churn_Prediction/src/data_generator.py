import numpy as np
import pandas as pd
import os

def create_customer_dataset(n_samples=2500, random_state=42, output_path="data/customer_churn_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(random_state)
    
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    tenure = np.random.exponential(scale=24, size=n_samples).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(loc=65, scale=25, size=n_samples).clip(18, 120).round(2)
    total_charges = (tenure * monthly_charges * np.random.uniform(0.95, 1.05, size=n_samples)).round(2)
    
    contract_types = np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], size=n_samples, p=[0.55, 0.25, 0.20])
    tech_support = np.random.choice(["Yes", "No"], size=n_samples, p=[0.4, 0.6])
    online_security = np.random.choice(["Yes", "No"], size=n_samples, p=[0.45, 0.55])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])
    
    # Calculate churn risk log-odds
    log_odds = (
        -1.2
        - 0.05 * tenure
        + 0.02 * monthly_charges
        + (contract_types == "Month-to-Month") * 0.9
        - (contract_types == "Two-Year") * 1.1
        - (tech_support == "Yes") * 0.7
        - (online_security == "Yes") * 0.5
    )
    probs = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.rand(n_samples) < probs).astype(int)
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_types,
        "tech_support": tech_support,
        "online_security": online_security,
        "paperless_billing": paperless_billing,
        "churn": churn
    })
    
    df.to_csv(output_path, index=False)
    return df
