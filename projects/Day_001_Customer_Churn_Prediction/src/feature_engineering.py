import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def build_features(df):
    df = df.copy()
    # Feature Engineering
    df["charge_per_tenure"] = df["monthly_charges"] / (df["tenure_months"] + 1)
    df["is_long_term"] = (df["tenure_months"] >= 24).astype(int)
    
    feature_cols = [
        "tenure_months", "monthly_charges", "total_charges",
        "charge_per_tenure", "is_long_term", "contract_type",
        "tech_support", "online_security", "paperless_billing"
    ]
    
    X = df[feature_cols]
    y = df["churn"]
    
    num_cols = ["tenure_months", "monthly_charges", "total_charges", "charge_per_tenure"]
    cat_cols = ["contract_type", "tech_support", "online_security", "paperless_billing"]
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
    ])
    
    return X, y, preprocessor
