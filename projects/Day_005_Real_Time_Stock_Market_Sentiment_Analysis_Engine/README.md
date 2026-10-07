# Day 5: Real-Time Stock Market Sentiment & Financial News Analytics Engine

![Domain](https://img.shields.io/badge/Domain-NLP-purple)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Market prices move rapidly based on earnings reports, corporate filings, and breaking news headlines. Inspired by algorithmic trading sentiment engines, this project implements:
1. Multi-ticker financial news stream generation (AAPL, MSFT, NVDA, TSLA, AMZN).
2. Domain-adapted financial sentiment extraction using VADER (Valence Aware Dictionary and sEntiment Reasoner).
3. Aggregate ticker polarity calculation: Positive, Neutral, Negative compound distributions.
4. Sentiment-to-Market signal classification (Bullish, Neutral, Bearish sentiment score).
5. Cross-ticker comparative polarity indexing and distribution plots.

## 🛠️ Project Structure
```text
Day_005_Real_Time_Stock_Market_Sentiment_Analysis_Engine/
├── data/
│   └── financial_news_feed.csv
├── results/
│   ├── ticker_sentiment_comparison.png
│   ├── sentiment_distribution.png
│   └── market_sentiment_summary.json
├── src/
│   ├── __init__.py
│   ├── news_data_generator.py
│   └── sentiment_analyzer.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_005_Real_Time_Stock_Market_Sentiment_Analysis_Engine
pip install -r requirements.txt
python main.py
```
