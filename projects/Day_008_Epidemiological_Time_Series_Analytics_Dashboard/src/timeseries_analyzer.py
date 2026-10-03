import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def analyze_epidemic_timeseries(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    df["date"] = pd.to_datetime(df["date"])
    
    # 7-day trailing moving average
    df["ma_7_cases"] = df["daily_reported_cases"].rolling(window=7, min_periods=1).mean()
    df["ma_14_cases"] = df["daily_reported_cases"].rolling(window=14, min_periods=1).mean()
    
    # Proxy Rt: ratio of 7-day MA current vs 7 days prior
    df["rt_proxy"] = df["ma_7_cases"] / (df["ma_7_cases"].shift(7) + 1e-9)
    df["rt_proxy"] = df["rt_proxy"].clip(0.2, 3.0)
    
    # 1. Plot Moving Averages
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["daily_reported_cases"], color="lightgray", alpha=0.8, label="Daily Reported (Raw)")
    plt.plot(df["date"], df["ma_7_cases"], color="#1f77b4", lw=2, label="7-Day Moving Average")
    plt.plot(df["date"], df["ma_14_cases"], color="#d62728", lw=1.8, linestyle="--", label="14-Day Trend")
    plt.title("Regional Epidemiological Daily Incidence & Moving Averages")
    plt.xlabel("Timeline")
    plt.ylabel("Reported Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "incidence_moving_averages.png"), dpi=200)
    plt.close()
    
    # 2. Plot Reproduction Rate (Rt Proxy)
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"][7:], df["rt_proxy"][7:], color="#ff7f0e", lw=1.8, label="Transmission Index (Rt)")
    plt.axhline(1.0, color="crimson", linestyle="--", label="Epidemic Equilibrium (Rt = 1.0)")
    plt.title("Effective Transmission Momentum (Rt Rate Proxy)")
    plt.xlabel("Timeline")
    plt.ylabel("Estimated Rt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "reproduction_rate_trend.png"), dpi=200)
    plt.close()
    
    peak_date = df.loc[df["ma_7_cases"].idxmax(), "date"].strftime("%Y-%m-%d")
    peak_count = int(df["ma_7_cases"].max())
    
    kpis = {
        "analysis_period_days": len(df),
        "total_cumulative_cases": int(df["daily_reported_cases"].sum()),
        "total_cumulative_recoveries": int(df["daily_recoveries"].sum()),
        "peak_incidence_date": peak_date,
        "peak_7day_incidence_avg": peak_count,
        "current_rt_momentum": round(float(df["rt_proxy"].iloc[-1]), 3)
    }
    
    with open(os.path.join(results_dir, "health_kpi_scorecard.json"), "w") as f:
        json.dump(kpis, f, indent=4)
        
    return kpis
