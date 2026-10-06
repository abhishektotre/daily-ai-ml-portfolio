# Day 4: Odontometric Biometric Gender Classification with Deep Neural Networks

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
Day_004_Odontometric_Biometric_Gender_Classification_Deep_Learning/
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
cd Day_004_Odontometric_Biometric_Gender_Classification_Deep_Learning
pip install -r requirements.txt
python main.py
```
