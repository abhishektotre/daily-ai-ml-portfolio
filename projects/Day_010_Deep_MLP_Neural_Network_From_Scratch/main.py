import os
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
