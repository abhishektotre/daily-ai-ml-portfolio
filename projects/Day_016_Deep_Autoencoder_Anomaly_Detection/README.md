# Day 16: Deep Autoencoder for Unsupervised Anomaly Detection

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
Day_016_Deep_Autoencoder_Anomaly_Detection/
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
cd Day_016_Deep_Autoencoder_Anomaly_Detection
pip install -r requirements.txt
python main.py
```
