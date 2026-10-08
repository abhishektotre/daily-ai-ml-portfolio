import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

class ExpenseAssistantAgent:
    def __init__(self, monthly_budget_limits=None):
        self.limits = monthly_budget_limits or {
            "Dining": 600.0,
            "Groceries": 1500.0,
            "Transportation": 800.0,
            "Infrastructure": 4000.0,
            "Travel": 3000.0,
            "Subscriptions": 200.0,
            "Healthcare": 500.0,
            "Software": 400.0,
            "Workspace": 2000.0
        }
        
    def process_and_audit(self, df, results_dir="results"):
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. Detect Category Outliers (Amounts > 3 std deviations above category mean)
        cat_stats = df.groupby("raw_category")["amount"].agg(["mean", "std"]).reset_index()
        df = df.merge(cat_stats, on="raw_category", how="left")
        
        df["is_anomaly"] = df["amount"] > (df["mean"] + 2.5 * df["std"])
        
        # 2. Total Spend per Category
        cat_totals = df.groupby("raw_category")["amount"].sum().to_dict()
        
        # 3. Autonomous Policy Engine Actions
        actions = []
        for cat, total in cat_totals.items():
            limit = self.limits.get(cat, 1000.0)
            utilization = (total / limit) * 100
            
            if utilization > 115.0:
                action = {
                    "category": cat,
                    "status": "CRITICAL_OVERSPEND",
                    "total_spent": round(total, 2),
                    "budget_limit": limit,
                    "utilization_pct": round(utilization, 1),
                    "recommendation": f"Freeze non-essential {cat} spend immediately and rebalance from surplus categories."
                }
            elif utilization > 90.0:
                action = {
                    "category": cat,
                    "status": "WARNING_HIGH_USAGE",
                    "total_spent": round(total, 2),
                    "budget_limit": limit,
                    "utilization_pct": round(utilization, 1),
                    "recommendation": f"Monitor {cat} closely; approaching budget threshold."
                }
            else:
                action = {
                    "category": cat,
                    "status": "HEALTHY",
                    "total_spent": round(total, 2),
                    "budget_limit": limit,
                    "utilization_pct": round(utilization, 1),
                    "recommendation": "Operating within nominal budget envelope."
                }
            actions.append(action)
            
        # Visualizations
        plt.figure(figsize=(7, 7))
        plt.pie(
            cat_totals.values(),
            labels=cat_totals.keys(),
            autopct="%1.1f%%",
            startangle=140,
            colors=sns.color_palette("Set2")
        )
        plt.title("Expense Distribution by Category")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "category_spend_pie.png"), dpi=200)
        plt.close()
        
        # Timeline
        df["date"] = pd.to_datetime(df["date"])
        daily_spend = df.groupby("date")["amount"].sum()
        
        plt.figure(figsize=(10, 4))
        daily_spend.plot(color="#2ca02c", lw=1.5)
        plt.title("Daily Expenditure Run-Rate")
        plt.xlabel("Date")
        plt.ylabel("Total Spend ($)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "spending_timeline.png"), dpi=200)
        plt.close()
        
        audit_report = {
            "total_expenses_processed": len(df),
            "total_spend_usd": round(float(df["amount"].sum()), 2),
            "anomalies_flagged_count": int(df["is_anomaly"].sum()),
            "category_audit_decisions": actions
        }
        
        with open(os.path.join(results_dir, "agent_workflow_audit.json"), "w") as f:
            json.dump(audit_report, f, indent=4)
            
        return audit_report
