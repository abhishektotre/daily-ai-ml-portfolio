import numpy as np
import pandas as pd
import os

def create_transaction_ledger(n_transactions=5000, n_customers=1000, output_path="data/customer_transactions.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    # Skewed customer distribution: top customers order much more frequently
    customer_weights = 1.0 / np.arange(1, n_customers + 1) ** 0.8
    customer_weights /= customer_weights.sum()
    
    chosen_customers = np.random.choice([f"CUST-{i:04d}" for i in range(1, n_customers + 1)], size=n_transactions, p=customer_weights)
    categories = ["Electronics", "Apparel", "Home & Kitchen", "Books", "Beauty", "Sports"]
    
    items_count = np.random.geometric(p=0.45, size=n_transactions).clip(1, 10)
    category_list = np.random.choice(categories, size=n_transactions, p=[0.25, 0.22, 0.20, 0.12, 0.11, 0.10])
    
    price_scales = {"Electronics": 180.0, "Apparel": 45.0, "Home & Kitchen": 65.0, "Books": 18.0, "Beauty": 32.0, "Sports": 55.0}
    scales = np.array([price_scales[c] for c in category_list])
    
    order_values = (scales * items_count * np.random.uniform(0.75, 1.25, size=n_transactions)).round(2)
    
    df = pd.DataFrame({
        "transaction_id": [f"TRX-{i:06d}" for i in range(1, n_transactions + 1)],
        "customer_id": chosen_customers,
        "category": category_list,
        "item_count": items_count,
        "revenue_usd": order_values
    })
    df.to_csv(output_path, index=False)
    return df
