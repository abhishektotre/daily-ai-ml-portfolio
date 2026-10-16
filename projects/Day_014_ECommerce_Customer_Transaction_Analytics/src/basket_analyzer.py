import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_transactions(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Customer Revenue Pareto Analysis
    cust_rev = df.groupby("customer_id")["revenue_usd"].sum().sort_values(ascending=False).reset_index()
    cust_rev["cum_revenue"] = cust_rev["revenue_usd"].cumsum()
    cust_rev["cum_rev_pct"] = cust_rev["cum_revenue"] / cust_rev["revenue_usd"].sum() * 100
    cust_rev["cum_cust_pct"] = (cust_rev.index + 1) / len(cust_rev) * 100
    
    # Pareto Plot (Lorenz Curve)
    plt.figure(figsize=(7, 5))
    plt.plot(cust_rev["cum_cust_pct"], cust_rev["cum_rev_pct"], color="#1f77b4", lw=2, label="Revenue Concentration")
    plt.plot([0, 100], [0, 100], color="gray", linestyle="--", label="Equal Distribution")
    plt.axvline(20, color="crimson", linestyle=":", label="Top 20% Customers")
    # Mark top 20% revenue share
    top_20_rev = cust_rev.loc[(cust_rev["cum_cust_pct"] - 20).abs().idxmin(), "cum_rev_pct"]
    plt.scatter([20], [top_20_rev], color="crimson", zorder=5)
    plt.annotate(f"{top_20_rev:.1f}% Revenue", xy=(20, top_20_rev), xytext=(28, top_20_rev - 8),
                 arrowprops=dict(facecolor="crimson", shrink=0.05, width=1, headwidth=6))
    plt.title("Pareto Revenue Curve (Cumulative Customer Share vs Revenue)")
    plt.xlabel("Cumulative Customer %")
    plt.ylabel("Cumulative Revenue %")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pareto_revenue_curve.png"), dpi=200)
    plt.close()
    
    # 2. Basket Size Distribution
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="item_count", palette="Blues_r")
    plt.title("Order Basket Size Distribution (Units per Order)")
    plt.xlabel("Number of Items in Basket")
    plt.ylabel("Transaction Count")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "basket_size_distribution.png"), dpi=200)
    plt.close()
    
    kpis = {
        "total_transactions": len(df),
        "unique_customers": int(df["customer_id"].nunique()),
        "gross_merchandise_value_usd": round(float(df["revenue_usd"].sum()), 2),
        "average_order_value_usd": round(float(df["revenue_usd"].mean()), 2),
        "average_basket_item_count": round(float(df["item_count"].mean()), 2),
        "top_20_pct_customer_revenue_share": round(float(top_20_rev), 2)
    }
    
    with open(os.path.join(results_dir, "transaction_kpis.json"), "w") as f:
        json.dump(kpis, f, indent=4)
        
    return kpis
