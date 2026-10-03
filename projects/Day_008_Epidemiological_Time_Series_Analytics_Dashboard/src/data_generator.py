import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def create_epidemic_data(n_days=180, output_path="data/regional_epidemic_trends.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    start_date = datetime(2025, 6, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Simulate multi-wave Gaussian bell curve profiles
    t = np.arange(n_days)
    wave1 = 650 * np.exp(-((t - 45) ** 2) / (2 * 14 ** 2))
    wave2 = 1100 * np.exp(-((t - 130) ** 2) / (2 * 20 ** 2))
    baseline = 40 + 0.1 * t
    daily_cases = baseline + wave1 + wave2 + np.random.normal(0, 25, size=n_days)
    daily_cases = np.maximum(daily_cases, 5).round().astype(int)
    
    # Reporting weekend lag: Sunday / Monday drops
    dow = np.array([d.weekday() for d in dates])
    daily_cases = np.where(dow == 6, daily_cases * 0.65, daily_cases).astype(int)
    daily_cases = np.where(dow == 0, daily_cases * 1.35, daily_cases).astype(int)
    
    # Recoveries lag cases by 14 days
    recovered = np.roll(daily_cases * 0.92, 14)
    recovered[:14] = daily_cases[:14] * 0.85
    recovered = np.maximum(recovered, 0).round().astype(int)
    
    df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "daily_reported_cases": daily_cases,
        "daily_recoveries": recovered,
        "active_hospitalizations": (daily_cases * 0.08 + np.random.normal(0, 5, n_days)).clip(2, 200).astype(int)
    })
    df.to_csv(output_path, index=False)
    return df
