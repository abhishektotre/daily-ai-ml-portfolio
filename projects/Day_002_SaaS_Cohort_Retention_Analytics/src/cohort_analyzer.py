import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_cohorts(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    
    # 1. Invoice Month & Cohort Month
    df["invoice_month"] = df["invoice_date"].dt.to_period("M")
    df["cohort_month"] = df.groupby("user_id")["invoice_month"].transform("min")
    
    # Cohort Index (integer distance in months)
    year_diff = df["invoice_month"].dt.year - df["cohort_month"].dt.year
    month_diff = df["invoice_month"].dt.month - df["cohort_month"].dt.month
    df["cohort_index"] = year_diff * 12 + month_diff
    
    # 2. Pivot Table of Active Users
    cohort_data = df.groupby(["cohort_month", "cohort_index"])["user_id"].apply(pd.Series.nunique).reset_index()
    cohort_counts = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="user_id")
    
    # Retention Matrix (Percentage)
    cohort_sizes = cohort_counts.iloc[:, 0]
    retention_matrix = cohort_counts.divide(cohort_sizes, axis=0) * 100
    
    # Plot Heatmap
    plt.figure(figsize=(11, 7))
    sns.heatmap(
        retention_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Retention Rate (%)"}
    )
    plt.title("SaaS Monthly Cohort Retention Heatmap (%)", fontsize=14, pad=12)
    plt.xlabel("Cohort Period (Months Since Signup)", fontsize=11)
    plt.ylabel("Acquisition Cohort Month", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cohort_retention_heatmap.png"), dpi=200)
    plt.close()
    
    # Plot Monthly Active Users (MAU)
    mau = df.groupby("invoice_month")["user_id"].nunique()
    plt.figure(figsize=(9, 4))
    mau.plot(kind="bar", color="#2b5c8f")
    plt.title("Monthly Active Customers (MAU)")
    plt.ylabel("Unique Paying Customers")
    plt.xlabel("Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "monthly_active_users.png"), dpi=200)
    plt.close()
    
    # Summary Metrics
    m1_retention = float(retention_matrix[1].dropna().mean()) if 1 in retention_matrix else 0.0
    m3_retention = float(retention_matrix[3].dropna().mean()) if 3 in retention_matrix else 0.0
    m6_retention = float(retention_matrix[6].dropna().mean()) if 6 in retention_matrix else 0.0
    
    summary = {
        "total_unique_customers": int(df["user_id"].nunique()),
        "total_revenue_generated": round(float(df["amount_paid"].sum()), 2),
        "avg_month_1_retention_pct": round(m1_retention, 2),
        "avg_month_3_retention_pct": round(m3_retention, 2),
        "avg_month_6_retention_pct": round(m6_retention, 2)
    }
    
    with open(os.path.join(results_dir, "cohort_metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    return summary
