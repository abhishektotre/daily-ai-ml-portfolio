import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_rfm_analysis(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    df["order_date"] = pd.to_datetime(df["order_date"])
    ref_date = df["order_date"].max() + pd.Timedelta(days=1)
    
    # Calculate RFM
    rfm = df.groupby("customer_id").agg({
        "order_date": lambda d: (ref_date - d.max()).days,
        "order_id": "count",
        "order_amount": "sum"
    }).rename(columns={
        "order_date": "recency",
        "order_id": "frequency",
        "order_amount": "monetary"
    })
    
    # Quantile Scoring
    rfm["R_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
    
    def assign_segment(row):
        score = row["R_score"] + row["F_score"] + row["M_score"]
        if score >= 10:
            return "Champions"
        elif score >= 8:
            return "Loyal Customers"
        elif score >= 6:
            return "Potential Loyalists"
        elif score >= 4:
            return "At Risk"
        else:
            return "Lost"
            
    rfm["segment"] = rfm.apply(assign_segment, axis=1)
    
    # Plot Segment Distribution
    plt.figure(figsize=(8, 4))
    segment_counts = rfm["segment"].value_counts()
    segment_counts.plot(kind="barh", color="#1f77b4")
    plt.title("Customer Segment Breakdown (RFM Framework)")
    plt.xlabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "segment_distribution.png"), dpi=200)
    plt.close()
    
    # Plot RFM Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=rfm,
        x="recency",
        y="monetary",
        hue="segment",
        palette="tab10",
        alpha=0.7
    )
    plt.title("Recency vs Monetary by Customer Segment")
    plt.xlabel("Recency (Days Since Last Order)")
    plt.ylabel("Total Spend ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "rfm_segments_scatter.png"), dpi=200)
    plt.close()
    
    summary = {
        "total_customers": len(rfm),
        "avg_recency_days": round(float(rfm["recency"].mean()), 1),
        "avg_orders_per_customer": round(float(rfm["frequency"].mean()), 2),
        "avg_customer_value": round(float(rfm["monetary"].mean()), 2),
        "segments": segment_counts.to_dict()
    }
    
    with open(os.path.join(results_dir, "segment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    return summary
