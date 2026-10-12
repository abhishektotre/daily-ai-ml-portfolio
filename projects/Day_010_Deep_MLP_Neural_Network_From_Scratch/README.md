# Day 10: Deep Multi-Layer Perceptron (MLP) with Adam & Backprop from Scratch

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
Day_010_Deep_MLP_Neural_Network_From_Scratch/
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
cd Day_010_Deep_MLP_Neural_Network_From_Scratch
pip install -r requirements.txt
python main.py
```
