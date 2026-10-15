import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def compute_kaplan_meier(durations, events):
    # Sort distinct event times
    df_temp = pd.DataFrame({"t": durations, "e": events}).sort_values("t")
    timeline = np.unique(df_temp["t"])
    
    n_at_risk = len(df_temp)
    surv_prob = 1.0
    
    curve_t = [0]
    curve_s = [1.0]
    
    for t in timeline:
        d = df_temp[(df_temp["t"] == t) & (df_temp["e"] == 1)].shape[0]
        c = df_temp[(df_temp["t"] == t) & (df_temp["e"] == 0)].shape[0]
        
        if n_at_risk > 0:
            surv_prob *= (1.0 - d / n_at_risk)
            curve_t.append(t)
            curve_s.append(surv_prob)
            n_at_risk -= (d + c)
            
    return np.array(curve_t), np.array(curve_s)

def run_survival_analysis(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    ctrl = df[df["treatment_group"] == "Control (Standard)"]
    trt = df[df["treatment_group"] == "Treatment (Novel Therapy)"]
    
    t_ctrl, s_ctrl = compute_kaplan_meier(ctrl["duration_days"].values, ctrl["event_occurred"].values)
    t_trt, s_trt = compute_kaplan_meier(trt["duration_days"].values, trt["event_occurred"].values)
    
    # Plot Kaplan-Meier Curves
    plt.figure(figsize=(8, 5))
    plt.step(t_ctrl, s_ctrl, where="post", label="Control (Standard)", color="#d62728", lw=2)
    plt.step(t_trt, s_trt, where="post", label="Treatment (Novel Therapy)", color="#2ca02c", lw=2)
    plt.title("Kaplan-Meier Survival Functions by Cohort")
    plt.xlabel("Time in Days")
    plt.ylabel("Estimated Survival Probability S(t)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "kaplan_meier_curves.png"), dpi=200)
    plt.close()
    
    # Cumulative Hazard: H(t) = -ln(S(t))
    h_ctrl = -np.log(np.clip(s_ctrl, 1e-9, 1.0))
    h_trt = -np.log(np.clip(s_trt, 1e-9, 1.0))
    
    plt.figure(figsize=(8, 4))
    plt.step(t_ctrl, h_ctrl, where="post", label="Control Hazard", color="#d62728", lw=1.8)
    plt.step(t_trt, h_trt, where="post", label="Treatment Hazard", color="#2ca02c", lw=1.8)
    plt.title("Nelson-Aalen Cumulative Hazard Function H(t)")
    plt.xlabel("Time in Days")
    plt.ylabel("Cumulative Hazard")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cumulative_hazard.png"), dpi=200)
    plt.close()
    
    # Median survival calculation
    def get_median(t, s):
        idx = np.where(s <= 0.5)[0]
        return float(t[idx[0]]) if len(idx) > 0 else "> 900 days"
        
    metrics = {
        "total_cohort_size": len(df),
        "censoring_rate_pct": round(float((1 - df["event_occurred"].mean()) * 100), 2),
        "median_survival_control_days": get_median(t_ctrl, s_ctrl),
        "median_survival_treatment_days": get_median(t_trt, s_trt),
        "survival_at_day_365_control": round(float(s_ctrl[np.searchsorted(t_ctrl, 365) - 1]), 4),
        "survival_at_day_365_treatment": round(float(s_trt[np.searchsorted(t_trt, 365) - 1]), 4)
    }
    
    with open(os.path.join(results_dir, "survival_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics
