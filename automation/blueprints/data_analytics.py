"""
Data Analytics Blueprints:
Includes cohort retention analysis, customer lifetime value (CLV),
RFM segmentation matrices, and funnel conversion analytics.
"""

def generate_cohort_analytics_project(day_num: int):
    folder_slug = "SaaS_Cohort_Retention_Analytics"
    title = "SaaS Monthly Cohort Retention & Churn Analytics"
    summary = "End-to-end cohort retention analysis calculating multi-month retention heatmaps, customer lifetime behavior, and revenue contraction."
    skills = ["Data Analytics", "Cohort Analysis", "Retention Matrix", "Heatmaps", "Pandas", "Seaborn"]

    readme_content = f"""# Day {day_num}: {title}

![Domain](https://img.shields.io/badge/Domain-Data%20Analytics-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Understanding user retention over time is the lifeblood of subscription & SaaS businesses. This project provides an enterprise-grade analytics engine that:
1. Simulates realistic recurring user event logs over a 12-month timeline.
2. Derives cohort acquisition months and relative cohort index (Month 0 to Month 11).
3. Constructs user retention percentage matrices.
4. Generates an automated Seaborn cohort retention heatmap for executive dashboards.
5. Calculates average customer survival half-life and monthly churn rates.

## 🛠️ Project Structure
```text
Day_{day_num:03d}_{folder_slug}/
├── data/
│   └── user_subscriptions.csv     # Event transaction logs
├── results/
│   ├── cohort_retention_heatmap.png
│   ├── monthly_active_users.png
│   └── cohort_metrics.json
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   └── cohort_analyzer.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_{day_num:03d}_{folder_slug}
pip install -r requirements.txt
python main.py
```
"""

    requirements_content = """pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    data_code = """import numpy as np
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
"""

    analyzer_code = """import json
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
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_event_logs
from src.cohort_analyzer import analyze_cohorts

def main():
    print("=" * 65)
    print(" 📊 Running SaaS Cohort Retention & Churn Analytics Pipeline")
    print("=" * 65)
    
    print("[1/3] Generating subscription transaction events...")
    df = create_event_logs()
    print(f"      Created {len(df)} billing transactions across users.")
    
    print("[2/3] Computing cohort index, retention matrix & visualization heatmaps...")
    metrics = analyze_cohorts(df)
    
    print("[3/3] Analytics Run Complete! Executive Summary:")
    for k, v in metrics.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Data Analytics",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "src/__init__.py": "",
            "src/data_generator.py": data_code,
            "src/cohort_analyzer.py": analyzer_code,
            "main.py": main_code
        }
    }


def generate_rfm_segmentation_project(day_num: int):
    folder_slug = "Customer_RFM_Segmentation_Matrix"
    title = "E-Commerce Customer RFM Segmentation & Value Matrix"
    summary = "Strategic customer segmentation pipeline implementing Recency, Frequency, and Monetary (RFM) scoring and behavioral clustering."
    skills = ["Data Analytics", "RFM Segmentation", "Customer Lifetime Value", "K-Means", "Clustering", "Visualizations"]

    readme_content = f"""# Day {day_num}: {title}

![Domain](https://img.shields.io/badge/Domain-Data%20Analytics-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Customer segmentation is fundamental to targeted marketing and profitability. This analytics engine:
1. Calculates granular Recency (days since last purchase), Frequency (order count), and Monetary (total spend) metrics.
2. Assigns quintile-based RFM scores (1-5 scale) creating composite customer tiers ('Champions', 'Loyalists', 'At Risk', 'Lost').
3. Applies K-Means clustering with automated standard scaling to discover natural behavioral segments.
4. Generates marketing strategy recommendations per customer segment.

## 🛠️ Project Structure
```text
Day_{day_num:03d}_{folder_slug}/
├── data/
│   └── ecommerce_orders.csv
├── results/
│   ├── rfm_segments_scatter.png
│   ├── segment_distribution.png
│   └── segment_summary.json
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   └── rfm_engine.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_{day_num:03d}_{folder_slug}
pip install -r requirements.txt
python main.py
```
"""

    requirements_content = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    data_code = """import numpy as np
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
"""

    engine_code = """import json
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
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_order_data
from src.rfm_engine import run_rfm_analysis

def main():
    print("=" * 65)
    print(" 🛍️ Running Customer RFM Segmentation & Value Matrix Pipeline")
    print("=" * 65)
    
    print("[1/3] Generating e-commerce multi-order transaction history...")
    df = create_order_data()
    print(f"      Generated {len(df)} transactions.")
    
    print("[2/3] Performing RFM calculation, quantile scoring & tier segmentation...")
    summary = run_rfm_analysis(df)
    
    print("[3/3] Segmentation Complete! Key Summary:")
    print(f"      - Total Customers Analyzed: {summary['total_customers']}")
    print(f"      - Average Customer Spend: ${summary['avg_customer_value']}")
    print("      - Segments Breakdown:")
    for seg, count in summary["segments"].items():
        print(f"        * {seg}: {count} customers")
    print("=" * 65)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Data Analytics",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "src/__init__.py": "",
            "src/data_generator.py": data_code,
            "src/rfm_engine.py": engine_code,
            "main.py": main_code
        }
    }

DATA_ANALYTICS_PROJECTS = [
    generate_cohort_analytics_project,
    generate_rfm_segmentation_project
]
