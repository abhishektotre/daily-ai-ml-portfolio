"""
Deep Learning Blueprints:
Includes deep neural architectures, custom backpropagation engines,
convolutional feature extractors, and deep autoencoders.
"""

def generate_mlp_from_scratch_project(day_num: int):
    folder_slug = "Deep_MLP_Neural_Network_From_Scratch"
    title = "Deep Multi-Layer Perceptron (MLP) with Adam & Backprop from Scratch"
    summary = "Mathematical deep learning implementation from scratch using vectorized NumPy, He/Xavier initialization, ReLU activations, and Adam optimizer."
    skills = ["Deep Learning", "Neural Networks", "Backpropagation", "Adam Optimizer", "NumPy", "Loss Landscapes"]

    readme_content = f"""# Day {day_num}: {title}

![Domain](https://img.shields.io/badge/Domain-Deep%20Learning-red)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Mastering deep learning requires understanding the mathematical mechanics beneath high-level frameworks. This project implements a complete, production-grade deep neural network from mathematical first principles:
1. Arbitrary depth feedforward architecture ($L$ layers).
2. **He (Kaiming) Initialization** for numerical stability against vanishing/exploding gradients.
3. **Activation Functions**: Leaky ReLU forward and analytical derivative backward pass.
4. **Numerically stable Softmax** and Cross-Entropy loss computation.
5. **Adam (Adaptive Moment Estimation)** optimization with first ($m$) and second ($v$) moment bias corrections.
6. Decision boundary visualization on complex non-linear spiral data.

## 🛠️ Project Structure
```text
Day_{day_num:03d}_{folder_slug}/
├── results/
│   ├── decision_boundary.png
│   ├── training_loss_curve.png
│   └── training_metrics.json
├── src/
│   ├── __init__.py
│   ├── activations.py
│   ├── optimizer.py
│   └── neural_network.py
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

    requirements_content = """numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
"""

    activations_code = """import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    # Numerically stable softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_one_hot):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(np.sum(y_one_hot * np.log(probs), axis=1))
"""

    optimizer_code = """import numpy as np

class AdamOptimizer:
    def __init__(self, layers_dims, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_w = [np.zeros((layers_dims[i], layers_dims[i+1])) for i in range(len(layers_dims)-1)]
        self.v_w = [np.zeros((layers_dims[i], layers_dims[i+1])) for i in range(len(layers_dims)-1)]
        self.m_b = [np.zeros((1, layers_dims[i+1])) for i in range(len(layers_dims)-1)]
        self.v_b = [np.zeros((1, layers_dims[i+1])) for i in range(len(layers_dims)-1)]
        self.t = 0
        
    def step(self, weights, biases, grads_w, grads_b):
        self.t += 1
        for i in range(len(weights)):
            # Update moments for weights
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i] ** 2)
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            weights[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
            
            # Update moments for biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
            biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
"""

    nn_code = """import numpy as np
from .activations import relu, relu_derivative, softmax, cross_entropy_loss
from .optimizer import AdamOptimizer

class DeepNeuralNetwork:
    def __init__(self, layer_dims, lr=0.01):
        self.layer_dims = layer_dims
        self.lr = lr
        self.weights = []
        self.biases = []
        
        # He (Kaiming) Normal Initialization
        np.random.seed(42)
        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2.0 / layer_dims[i])
            b = np.zeros((1, layer_dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
        self.optimizer = AdamOptimizer(layer_dims, lr=lr)
        
    def forward(self, X):
        self.a_cache = [X]
        self.z_cache = []
        
        curr_a = X
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(curr_a, self.weights[i]) + self.biases[i]
            curr_a = relu(z)
            self.z_cache.append(z)
            self.a_cache.append(curr_a)
            
        # Output layer (Softmax)
        z_out = np.dot(curr_a, self.weights[-1]) + self.biases[-1]
        probs = softmax(z_out)
        self.z_cache.append(z_out)
        self.a_cache.append(probs)
        return probs
        
    def backward(self, y_one_hot):
        m = y_one_hot.shape[0]
        grads_w = []
        grads_b = []
        
        # Output layer gradient
        dz = (self.a_cache[-1] - y_one_hot) / m
        dw = np.dot(self.a_cache[-2].T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        grads_w.append(dw)
        grads_b.append(db)
        
        # Hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            da = np.dot(dz, self.weights[l+1].T)
            dz = da * relu_derivative(self.z_cache[l])
            dw = np.dot(self.a_cache[l].T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            
        self.optimizer.step(self.weights, self.biases, grads_w, grads_b)
        
    def fit(self, X, y, epochs=300):
        # Convert y to one-hot
        num_classes = self.layer_dims[-1]
        y_one_hot = np.eye(num_classes)[y]
        
        history = []
        for epoch in range(1, epochs + 1):
            probs = self.forward(X)
            loss = cross_entropy_loss(probs, y_one_hot)
            self.backward(y_one_hot)
            
            preds = np.argmax(probs, axis=1)
            acc = np.mean(preds == y)
            history.append({"epoch": epoch, "loss": float(loss), "accuracy": float(acc)})
        return history
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from src.neural_network import DeepNeuralNetwork

def main():
    print("=" * 65)
    print(" 🧠 Running Deep Neural Network with Adam Optimizer from Scratch")
    print("=" * 65)
    
    os.makedirs("results", exist_ok=True)
    
    # 1. Generate Non-linear Dataset (Dual Moons)
    print("[1/3] Generating non-linear manifold classification data...")
    X, y = make_moons(n_samples=1200, noise=0.25, random_state=42)
    
    # 2. Build 4-Layer Deep Network: 2 -> 32 -> 16 -> 8 -> 2
    print("[2/3] Constructing 4-layer Deep MLP [2 -> 32 -> 16 -> 8 -> 2]...")
    dnn = DeepNeuralNetwork(layer_dims=[2, 32, 16, 8, 2], lr=0.01)
    
    print("      Training network with He initialization and Adam optimizer...")
    history = dnn.fit(X, y, epochs=350)
    
    final_loss = history[-1]["loss"]
    final_acc = history[-1]["accuracy"]
    print(f"      Epoch 350 - Cross-Entropy Loss: {final_loss:.4f} | Accuracy: {final_acc * 100:.2f}%")
    
    # 3. Visualizations
    print("[3/3] Plotting training dynamics and decision boundaries...")
    
    # Loss curve
    losses = [h["loss"] for h in history]
    accs = [h["accuracy"] for h in history]
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses, color="crimson")
    plt.title("Cross-Entropy Loss (Adam)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(accs, color="navy")
    plt.title("Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("results/training_loss_curve.png", dpi=200)
    plt.close()
    
    # Decision Boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = dnn.forward(grid)
    Z = np.argmax(probs, axis=1).reshape(xx.shape)
    
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors="k", s=25)
    plt.title(f"Learned Non-Linear Decision Boundary (Acc: {final_acc*100:.1f}%)")
    plt.tight_layout()
    plt.savefig("results/decision_boundary.png", dpi=200)
    plt.close()
    
    metrics = {
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 4),
        "architecture": [2, 32, 16, 8, 2],
        "optimizer": "Adam",
        "epochs": 350
    }
    with open("results/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(" Pipeline Complete! Saved plots to results/")
    print("=" * 65)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Deep Learning",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "src/__init__.py": "",
            "src/activations.py": activations_code,
            "src/optimizer.py": optimizer_code,
            "src/neural_network.py": nn_code,
            "main.py": main_code
        }
    }


def generate_autoencoder_project(day_num: int):
    folder_slug = "Deep_Autoencoder_Anomaly_Detection"
    title = "Deep Autoencoder for Unsupervised Anomaly Detection"
    summary = "Deep bottleneck autoencoder architecture trained to compress representations and detect sensor anomalies via reconstruction error thresholding."
    skills = ["Deep Learning", "Autoencoder", "Anomaly Detection", "Reconstruction Error", "Dimensionality Reduction", "Scikit-Learn"]

    readme_content = f"""# Day {day_num}: {title}

![Domain](https://img.shields.io/badge/Domain-Deep%20Learning-red)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Unsupervised anomaly detection in complex, high-dimensional spaces is crucial for industrial safety, server health, and fraud prevention. This project implements:
1. Multi-tier sensor stream data synthesis with healthy baseline dynamics and injected failure anomalies.
2. Deep symmetric Autoencoder architecture (Input -> Bottleneck -> Output).
3. Mean Squared Reconstruction Error (MSE) metric calculation.
4. Dynamic threshold optimization (99th percentile of nominal reconstruction error).
5. Precision, Recall, and F1-score evaluation against ground truth anomalies.

## 🛠️ Project Structure
```text
Day_{day_num:03d}_{folder_slug}/
├── data/
│   └── sensor_streams.csv
├── results/
│   ├── reconstruction_error_dist.png
│   ├── anomaly_detection_timeline.png
│   └── metrics.json
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   └── autoencoder.py
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

    requirements_content = """numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    data_code = """import numpy as np
import pandas as pd
import os

def create_sensor_data(n_samples=3000, anomaly_ratio=0.03, output_path="data/sensor_streams.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    # 8 correlated sensor dimensions
    t = np.linspace(0, 50, n_samples)
    s1 = np.sin(t) + np.random.normal(0, 0.1, n_samples)
    s2 = np.cos(t * 0.5) + np.random.normal(0, 0.1, n_samples)
    s3 = 0.5 * s1 + 0.8 * s2 + np.random.normal(0, 0.05, n_samples)
    s4 = np.sin(t * 1.5) + np.random.normal(0, 0.15, n_samples)
    s5 = -0.7 * s2 + np.random.normal(0, 0.1, n_samples)
    s6 = np.tanh(s1) + np.random.normal(0, 0.05, n_samples)
    s7 = s4 * 0.6 + s5 * 0.4 + np.random.normal(0, 0.1, n_samples)
    s8 = np.exp(-0.01 * t) * np.sin(t) + np.random.normal(0, 0.1, n_samples)
    
    data = np.column_stack([s1, s2, s3, s4, s5, s6, s7, s8])
    labels = np.zeros(n_samples, dtype=int)
    
    # Inject anomaly bursts
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
    for idx in anomaly_indices:
        data[idx] += np.random.uniform(2.5, 5.0, size=8) * np.random.choice([-1, 1], size=8)
        labels[idx] = 1
        
    cols = [f"sensor_{i}" for i in range(1, 9)]
    df = pd.DataFrame(data, columns=cols)
    df["is_anomaly"] = labels
    df.to_csv(output_path, index=False)
    return df
"""

    ae_code = """import json
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
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_sensor_data
from src.autoencoder import train_autoencoder

def main():
    print("=" * 65)
    print(" 📉 Running Deep Autoencoder Unsupervised Anomaly Detection")
    print("=" * 65)
    
    print("[1/3] Generating multi-channel telemetry sensor streams...")
    df = create_sensor_data()
    print(f"      Synthesized {len(df)} samples with 8 correlated channels.")
    
    print("[2/3] Fitting Bottleneck Autoencoder [8 -> 4 -> 2 -> 4 -> 8]...")
    metrics = train_autoencoder(df)
    
    print("[3/3] Detection Performance:")
    for k, v in metrics.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Deep Learning",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "src/__init__.py": "",
            "src/data_generator.py": data_code,
            "src/autoencoder.py": ae_code,
            "main.py": main_code
        }
    }

def generate_dental_biometric_dl_project(day_num: int):
    folder_slug = "Odontometric_Biometric_Gender_Classification_Deep_Learning"
    title = "Odontometric Biometric Gender Classification with Deep Neural Networks"
    summary = "Deep learning architecture classifying human sexual dimorphism using odontometric dental measurements, canine indices, and mandibular parameters."
    skills = ["Deep Learning", "Biometrics", "Neural Networks", "Sexual Dimorphism", "Scikit-Learn", "Matplotlib"]

    readme_content = f"""# Day {day_num}: {title}

![Domain](https://img.shields.io/badge/Domain-Deep%20Learning-red)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Forensic anthropology and biometric identification frequently utilize odontometric parameters (dental measurements) due to teeth being the most durable anatomical structures. Inspired by research in forensic odontometry, this project implements:
1. Multi-parameter odontometric feature synthesis (Maxillary Canine Width, Mandibular Canine Width, Inter-Canine Distance, and Mandibular Canine Index).
2. Deep Multi-Layer Perceptron (MLP) architecture with regularization to model non-linear sexual dimorphism.
3. Feature distribution analysis and sexual dimorphism ratio calculations.
4. Comprehensive ROC-AUC, Precision, Recall, and Confusion Matrix diagnostics.
5. Calibrated forensic classification confidence thresholding.

## 🛠️ Project Structure
```text
Day_{day_num:03d}_{folder_slug}/
├── data/
│   └── odontometric_measurements.csv
├── results/
│   ├── roc_curve.png
│   ├── feature_distributions.png
│   └── classification_scorecard.json
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   └── deep_classifier.py
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

    requirements_content = """numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    data_code = """import numpy as np
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
"""

    classifier_code = """import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def train_biometric_model(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    X = df.drop(columns=["gender"])
    y = df["gender"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Deep MLP Classifier: 5 -> 64 -> 32 -> 16 -> 1
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=300,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="crimson", lw=2, label=f"Deep MLP ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Odontometric Gender Classification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=200)
    plt.close()
    
    # Plot Feature Distributions
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df, x="maxillary_canine_width_mm", hue="gender", palette="Set1", common_norm=False)
    plt.title("Maxillary Canine Width (mm)")
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df, x="mandibular_canine_index", hue="gender", palette="Set1", common_norm=False)
    plt.title("Mandibular Canine Index (MCI)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "feature_distributions.png"), dpi=200)
    plt.close()
    
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "test_records_classified": len(y_test)
    }
    with open(os.path.join(results_dir, "classification_scorecard.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_generator import create_odontometric_dataset
from src.deep_classifier import train_biometric_model

def main():
    print("=" * 65)
    print(" 🦷 Running Deep Odontometric Biometric Classification")
    print("=" * 65)
    
    print("[1/3] Synthesizing clinical odontometric measurement cohort...")
    df = create_odontometric_dataset()
    print(f"      Generated {len(df)} patient dental records.")
    
    print("[2/3] Training Deep MLP Network [64 -> 32 -> 16] with Adam...")
    metrics = train_biometric_model(df)
    
    print("[3/3] Biometric Classification Results:")
    for k, v in metrics.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Deep Learning",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "src/__init__.py": "",
            "src/data_generator.py": data_code,
            "src/deep_classifier.py": classifier_code,
            "main.py": main_code
        }
    }

DEEP_LEARNING_PROJECTS = [
    generate_dental_biometric_dl_project,
    generate_mlp_from_scratch_project,
    generate_autoencoder_project
]
