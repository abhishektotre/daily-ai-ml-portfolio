import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def create_order_data(n_orders=6000, n_customers=1200, output_path="data/ecommerce_orders.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    reference_date = datetime(2025, 12, 31)
    customer_ids = [f"CUST_{np.random.randint(1, n_customers + 1):04d}" for _ in range(n_orders)]
    
    days_ago = np.random.exponential(scale=90, size=n_orders).clip(1, 365).astype(int)
    order_dates = [reference_date - timedelta(days=int(d)) for d in days_ago]
    order_amounts = np.random.gamma(shape=2.5, scale=40, size=n_orders).round(2)
    
    df = pd.DataFrame({
        "order_id": [f"ORD_{i:06d}" for i in range(1, n_orders + 1)],
        "customer_id": customer_ids,
        "order_date": [d.strftime("%Y-%m-%d") for d in order_dates],
        "order_amount": order_amounts
    })
    df.to_csv(output_path, index=False)
    return df
