import numpy as np
import pandas as pd
import os

def create_survival_cohort(n_patients=1600, output_path="data/patient_survival_records.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    treatment_arm = np.random.choice(["Control (Standard)", "Treatment (Novel Therapy)"], size=n_patients, p=[0.5, 0.5])
    age = np.random.normal(loc=58, scale=9, size=n_patients).clip(30, 85).astype(int)
    
    # Treatment group experiences longer survival times
    scale_param = np.where(treatment_arm == "Control (Standard)", 450.0, 720.0)
    
    # Weibull time-to-event distribution (shape k=1.3)
    k = 1.3
    actual_survival_days = scale_param * np.random.weibull(k, size=n_patients)
    
    # Study duration cutoff (censorship) at 900 days
    study_duration = 900
    observed_time = np.minimum(actual_survival_days, study_duration).round(1)
    event_observed = (actual_survival_days <= study_duration).astype(int) # 1 = Event, 0 = Censored
    
    df = pd.DataFrame({
        "patient_id": [f"PT-{i:05d}" for i in range(1, n_patients + 1)],
        "age": age,
        "treatment_group": treatment_arm,
        "duration_days": observed_time,
        "event_occurred": event_observed
    })
    df.to_csv(output_path, index=False)
    return df
