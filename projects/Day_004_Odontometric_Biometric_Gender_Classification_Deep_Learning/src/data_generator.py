import numpy as np
import pandas as pd
import os

def create_odontometric_dataset(n_samples=2400, output_path="data/odontometric_measurements.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    n_per_class = n_samples // 2
    
    # Males exhibit greater canine dimensions (sexual dimorphism)
    # Canine width in mm
    male_max_canine = np.random.normal(loc=7.95, scale=0.48, size=n_per_class)
    female_max_canine = np.random.normal(loc=7.25, scale=0.45, size=n_per_class)
    
    male_mand_canine = np.random.normal(loc=6.98, scale=0.42, size=n_per_class)
    female_mand_canine = np.random.normal(loc=6.32, scale=0.40, size=n_per_class)
    
    male_icd = np.random.normal(loc=27.4, scale=1.8, size=n_per_class)
    female_icd = np.random.normal(loc=25.6, scale=1.6, size=n_per_class)
    
    # Mandibular Canine Index (MCI = Mandibular Canine Width / Inter-canine Distance)
    male_mci = male_mand_canine / male_icd
    female_mci = female_mand_canine / female_icd
    
    # Arch width
    male_arch = np.random.normal(loc=35.2, scale=2.1, size=n_per_class)
    female_arch = np.random.normal(loc=33.1, scale=1.9, size=n_per_class)
    
    mcw = np.concatenate([male_max_canine, female_max_canine])
    mnw = np.concatenate([male_mand_canine, female_mand_canine])
    icd = np.concatenate([male_icd, female_icd])
    mci = np.concatenate([male_mci, female_mci])
    arch = np.concatenate([male_arch, female_arch])
    gender = np.array([1] * n_per_class + [0] * n_per_class) # 1 = Male, 0 = Female
    
    idx = np.random.permutation(n_samples)
    
    df = pd.DataFrame({
        "maxillary_canine_width_mm": mcw[idx].round(2),
        "mandibular_canine_width_mm": mnw[idx].round(2),
        "inter_canine_distance_mm": icd[idx].round(2),
        "mandibular_canine_index": mci[idx].round(4),
        "dental_arch_width_mm": arch[idx].round(2),
        "gender": gender[idx]
    })
    df.to_csv(output_path, index=False)
    return df
