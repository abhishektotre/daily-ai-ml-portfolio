import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_expenses(n_records=900, output_path="data/expense_transactions.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    merchants = {
        "Starbucks Coffee": ("Dining", 4.5, 12.0),
        "Whole Foods Market": ("Groceries", 35.0, 180.0),
        "Uber Rides": ("Transportation", 15.0, 65.0),
        "AWS Cloud Services": ("Infrastructure", 120.0, 850.0),
        "Delta Airlines": ("Travel", 220.0, 950.0),
        "Netflix Subscription": ("Subscriptions", 15.49, 15.49),
        "CVS Pharmacy": ("Healthcare", 12.0, 95.0),
        "Chevron Gas": ("Transportation", 30.0, 75.0),
        "Apple App Store": ("Software", 2.99, 49.99),
        "WeWork Coworking": ("Workspace", 300.0, 550.0)
    }
    
    merchant_names = list(merchants.keys())
    base_date = datetime(2026, 1, 1)
    
    records = []
    for i in range(n_records):
        m_name = np.random.choice(merchant_names)
        category, min_amt, max_amt = merchants[m_name]
        
        # Random normal / uniform amount
        amt = np.random.uniform(min_amt, max_amt)
        
        # Inject occasional anomaly (huge unapproved expense)
        if np.random.rand() < 0.015:
            amt *= np.random.uniform(4.0, 8.0)
            
        timestamp = base_date + timedelta(hours=int(i * 3.2))
        
        records.append({
            "transaction_id": f"TX-{10000 + i}",
            "date": timestamp.strftime("%Y-%m-%d"),
            "merchant": m_name,
            "raw_category": category,
            "amount": round(float(amt), 2),
            "payment_method": np.random.choice(["Corporate Card", "Personal Card", "Wire Transfer"], p=[0.75, 0.20, 0.05])
        })
        
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return df
