import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

def train_autoencoder(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    X = df[sensor_cols].values
    y = df["is_anomaly"].values
    
    # Train only on nominal data (unsupervised)
    X_nominal = X[y == 0]
    
    scaler = StandardScaler()
    X_nominal_scaled = scaler.fit_transform(X_nominal)
    X_all_scaled = scaler.transform(X)
    
    # Deep Autoencoder: 8 -> 4 -> 2 (bottleneck) -> 4 -> 8
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(4, 2, 4),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42
    )
    autoencoder.fit(X_nominal_scaled, X_nominal_scaled)
    
    # Reconstruction
    X_reconstructed = autoencoder.predict(X_all_scaled)
    mse = np.mean(np.square(X_all_scaled - X_reconstructed), axis=1)
    
    # Nominal 99th percentile threshold
    nominal_mse = mse[y == 0]
    threshold = float(np.percentile(nominal_mse, 99))
    
    y_pred = (mse > threshold).astype(int)
    
    # Metrics
    f1 = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    
    # Plot Error Distribution
    plt.figure(figsize=(7, 4))
    sns.kdeplot(mse[y == 0], label="Nominal Data", fill=True, color="green")
    sns.kdeplot(mse[y == 1], label="Injected Anomalies", fill=True, color="red")
    plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.2f})")
    plt.title("Reconstruction Error Distribution (Nominal vs Anomaly)")
    plt.xlabel("Reconstruction MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "reconstruction_error_dist.png"), dpi=200)
    plt.close()
    
    # Plot Anomaly Timeline
    plt.figure(figsize=(10, 4))
    plt.plot(mse[:400], color="#2b5c8f", lw=1.2, label="Reconstruction MSE")
    plt.axhline(threshold, color="crimson", linestyle="--", label="Anomaly Threshold")
    anomaly_pts = np.where(y[:400] == 1)[0]
    plt.scatter(anomaly_pts, mse[anomaly_pts], color="red", s=35, zorder=5, label="True Anomaly")
    plt.title("Sensor Stream Anomaly Detection Window")
    plt.xlabel("Time Step")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "anomaly_detection_timeline.png"), dpi=200)
    plt.close()
    
    metrics = {
        "reconstruction_threshold": round(threshold, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "total_anomalies_flagged": int(np.sum(y_pred)),
        "true_anomalies_present": int(np.sum(y))
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics
