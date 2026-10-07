import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_market_sentiment(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    analyzer = SentimentIntensityAnalyzer()
    
    scores = []
    for text in df["headline"]:
        res = analyzer.polarity_scores(text)
        scores.append(res)
        
    score_df = pd.DataFrame(scores)
    df["compound"] = score_df["compound"]
    df["pos"] = score_df["pos"]
    df["neu"] = score_df["neu"]
    df["neg"] = score_df["neg"]
    
    def classify_sentiment(compound):
        if compound >= 0.05:
            return "Bullish"
        elif compound <= -0.05:
            return "Bearish"
        else:
            return "Neutral"
            
    df["sentiment_signal"] = df["compound"].apply(classify_sentiment)
    
    # 1. Ticker Mean Sentiment Comparison Plot
    ticker_agg = df.groupby("ticker")["compound"].mean().reset_index()
    
    plt.figure(figsize=(8, 4))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ticker_agg["compound"]]
    plt.bar(ticker_agg["ticker"], ticker_agg["compound"], color=colors)
    plt.axhline(0, color="gray", linestyle="--")
    plt.title("Net Sentiment Polarity Score by Ticker (VADER)")
    plt.xlabel("Ticker")
    plt.ylabel("Mean Compound Score (-1.0 to +1.0)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ticker_sentiment_comparison.png"), dpi=200)
    plt.close()
    
    # 2. Overall Sentiment Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="sentiment_signal", palette="coolwarm", order=["Bullish", "Neutral", "Bearish"])
    plt.title("Financial Headline Sentiment Distribution")
    plt.xlabel("Sentiment Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "sentiment_distribution.png"), dpi=200)
    plt.close()
    
    summary = {
        "total_headlines_analyzed": len(df),
        "mean_market_sentiment": round(float(df["compound"].mean()), 4),
        "sentiment_counts": df["sentiment_signal"].value_counts().to_dict(),
        "ticker_rankings": {row["ticker"]: round(float(row["compound"]), 4) for _, row in ticker_agg.iterrows()}
    }
    
    with open(os.path.join(results_dir, "market_sentiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    return summary
