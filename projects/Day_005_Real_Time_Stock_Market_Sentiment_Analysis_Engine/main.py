import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.news_data_generator import create_financial_news
from src.sentiment_analyzer import analyze_market_sentiment

def main():
    print("=" * 65)
    print(" 📈 Running Real-Time Stock Market Sentiment Analytics Engine")
    print("=" * 65)
    
    print("[1/3] Ingesting multi-ticker financial news stream...")
    df = create_financial_news()
    print(f"      Processed {len(df)} financial news headlines across {df['ticker'].nunique()} tickers.")
    
    print("[2/3] Computing VADER compound polarity & bullish/bearish signals...")
    summary = analyze_market_sentiment(df)
    
    print("[3/3] Financial Sentiment Scorecard:")
    print(f"      - Overall Market Sentiment Index: {summary['mean_market_sentiment']}")
    print(f"      - Signals Breakdown: {summary['sentiment_counts']}")
    print("      - Ticker Polarity Ranks:")
    for ticker, score in summary["ticker_rankings"].items():
        print(f"        * {ticker}: {score:+.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
