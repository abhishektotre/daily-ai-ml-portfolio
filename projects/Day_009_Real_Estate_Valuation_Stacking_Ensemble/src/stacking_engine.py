import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_stacking_ensemble(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    X = df.drop(columns=["price_usd"])
    y = df["price_usd"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    estimators = [
        ("ridge", Ridge(alpha=10.0)),
        ("rf", RandomForestRegressor(n_estimators=80, max_depth=7, random_state=42)),
        ("gbr", GradientBoostingRegressor(n_estimators=90, learning_rate=0.08, random_state=42))
    ]
    
    stacker = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5
    )
    
    stacker.fit(X_train, y_train)
    y_pred = stacker.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test / 1000, y_pred / 1000, alpha=0.5, color="#2b5c8f")
    lims = [min(y_test.min(), y_pred.min()) / 1000, max(y_test.max(), y_pred.max()) / 1000]
    plt.plot(lims, lims, color="red", linestyle="--", lw=2)
    plt.title(f"Actual vs Predicted Valuation ($k) - R2: {r2:.3f}")
    plt.xlabel("Actual Price ($k)")
    plt.ylabel("Predicted Price ($k)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "actual_vs_predicted.png"), dpi=200)
    plt.close()
    
    metrics = {
        "rmse_usd": round(float(rmse), 2),
        "mae_usd": round(float(mae), 2),
        "r2_score": round(float(r2), 4),
        "sample_count": len(y_test)
    }
    
    with open(os.path.join(results_dir, "regression_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics
