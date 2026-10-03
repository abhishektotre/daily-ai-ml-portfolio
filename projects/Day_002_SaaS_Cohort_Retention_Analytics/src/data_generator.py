import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def create_event_logs(n_users=1500, output_path="data/user_subscriptions.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    start_date = datetime(2025, 1, 1)
    records = []
    
    for user_id in range(1, n_users + 1):
        # Assign sign-up month (0 to 9)
        signup_month_idx = np.random.randint(0, 10)
        user_signup = start_date + timedelta(days=int(signup_month_idx * 30 + np.random.randint(0, 28)))
        
        # User retention decay parameter
        retention_decay = np.random.uniform(0.65, 0.90)
        
        current_date = user_signup
        month_offset = 0
        while month_offset < (12 - signup_month_idx):
            if month_offset == 0 or np.random.rand() < (retention_decay ** month_offset):
                mrr = np.random.choice([29.0, 79.0, 199.0], p=[0.6, 0.3, 0.1])
                records.append({
                    "user_id": f"USR_{user_id:05d}",
                    "invoice_date": current_date.strftime("%Y-%m-%d"),
                    "amount_paid": mrr,
                    "plan": "Pro" if mrr == 79 else ("Enterprise" if mrr == 199 else "Basic")
                })
            else:
                # Churned
                break
            month_offset += 1
            current_date += timedelta(days=30)
            
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return df
