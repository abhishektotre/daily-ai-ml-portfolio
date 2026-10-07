import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_financial_news(n_records=1200, output_path="data/financial_news_feed.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
    
    bullish_templates = [
        "{ticker} delivers blowout quarterly earnings, exceeding analyst EPS forecasts by 18%.",
        "Record-breaking demand for {ticker} next-generation AI chip architectures signals surging revenue.",
        "Major institutional upgrade lifts {ticker} price target by 25% following breakthrough product launch.",
        "{ticker} expands strategic cloud partnership, securing massive multi-year enterprise contracts.",
        "Positive margin expansion and robust cash flow momentum boost {ticker} shares in pre-market rally."
    ]
    
    neutral_templates = [
        "{ticker} announces upcoming annual shareholder conference scheduled for next quarter.",
        "{ticker} maintains fiscal guidance in line with consensus market expectations.",
        "Federal regulatory review underway regarding {ticker} latest acquisition filing.",
        "{ticker} management reshuffles operational team to focus on international expansion.",
        "Market analysts maintain hold rating on {ticker} pending release of inflation print."
    ]
    
    bearish_templates = [
        "{ticker} falls short of quarterly revenue estimates amid supply chain disruptions.",
        "Disappointing consumer demand forces {ticker} to lower full-year margin outlook.",
        "Severe antitrust scrutiny and regulatory headwinds pressure {ticker} operating margins.",
        "{ticker} issues cautious guidance warning of slowing global enterprise software spend.",
        "Unexpected executive departure and delayed product rollout spark selloff in {ticker}."
    ]
    
    records = []
    base_date = datetime(2026, 1, 1)
    
    for i in range(n_records):
        ticker = np.random.choice(tickers)
        bias = np.random.choice(["bullish", "neutral", "bearish"], p=[0.42, 0.30, 0.28])
        if bias == "bullish":
            headline = np.random.choice(bullish_templates).format(ticker=ticker)
        elif bias == "neutral":
            headline = np.random.choice(neutral_templates).format(ticker=ticker)
        else:
            headline = np.random.choice(bearish_templates).format(ticker=ticker)
            
        timestamp = base_date + timedelta(hours=int(i * 2.5))
        records.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "headline": headline
        })
        
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return df
