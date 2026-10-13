# Day 11: Multi-Aspect Customer Feedback Sentiment & Emotion Engine

![Domain](https://img.shields.io/badge/Domain-NLP-purple)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Analyzing customer feedback across dimensions (Product Quality, Shipping, Customer Support, Price) requires nuanced text analytics. This NLP system:
1. Synthesizes realistic e-commerce and SaaS product review corpora with labeled positive/neutral/negative sentiments.
2. Performs linguistic text normalization (lowercasing, stopword filtering, punctuation removal, n-gram tokenization).
3. Builds TF-IDF (Term Frequency-Inverse Document Frequency) sub-linear feature matrices.
4. Trains a Multinomial Naive Bayes text classification model with Laplace smoothing.
5. Performs aspect-level sentiment breakdown and extracts high-leverage polar keywords.

## 🛠️ Project Structure
```text
Day_011_Multi_Aspect_Sentiment_Analysis_Engine/
├── data/
│   └── reviews.csv
├── results/
│   ├── sentiment_confusion_matrix.png
│   ├── top_polar_keywords.png
│   └── nlp_metrics.json
├── src/
│   ├── __init__.py
│   ├── text_data_generator.py
│   └── sentiment_pipeline.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_011_Multi_Aspect_Sentiment_Analysis_Engine
pip install -r requirements.txt
python main.py
```
