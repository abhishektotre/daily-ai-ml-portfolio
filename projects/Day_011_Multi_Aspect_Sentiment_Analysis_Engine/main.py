import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.text_data_generator import create_review_dataset
from src.sentiment_pipeline import train_sentiment_model

def main():
    print("=" * 65)
    print(" 💬 Running NLP Pipeline: Multi-Aspect Sentiment Engine")
    print("=" * 65)
    
    print("[1/3] Generating synthetic multilingual customer reviews...")
    df = create_review_dataset()
    print(f"      Created dataset with {len(df)} customer reviews.")
    
    print("[2/3] Extracting TF-IDF n-grams & training Multinomial Naive Bayes...")
    metrics = train_sentiment_model(df)
    
    print("[3/3] NLP Model Results:")
    for k, v in metrics.items():
        print(f"      - {k}: {v}")
    print("=" * 65)

if __name__ == "__main__":
    main()
