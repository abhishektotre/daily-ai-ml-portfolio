import numpy as np
import pandas as pd
import os

def create_property_data(n_samples=2000, output_path="data/properties.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    sqft = np.random.normal(loc=2100, scale=600, size=n_samples).clip(800, 5000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.25, 0.45, 0.20, 0.05])
    bathrooms = np.clip(np.round(bedrooms * 0.75 + np.random.normal(0, 0.5, n_samples), 1), 1, 4.5)
    school_rating = np.random.randint(1, 11, size=n_samples) # 1-10
    dist_downtown = np.random.exponential(scale=8, size=n_samples).clip(0.5, 30) # km
    age_years = np.random.uniform(0, 60, size=n_samples)
    
    # Valuation function with non-linear factors
    base_price = (
        120000
        + 185 * sqft
        + 15000 * bedrooms
        + 22000 * bathrooms
        + 12000 * school_rating
        - 4500 * dist_downtown
        - 800 * age_years
        + np.random.normal(0, 25000, size=n_samples)
    ).clip(85000, 1500000)
    
    df = pd.DataFrame({
        "sqft_living": sqft.round(),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "school_rating": school_rating,
        "distance_downtown_km": dist_downtown.round(2),
        "age_years": age_years.round(1),
        "price_usd": base_price.round(-2)
    })
    df.to_csv(output_path, index=False)
    return df
