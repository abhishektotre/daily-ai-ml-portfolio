import pandas as pd
import numpy as np
import os

def create_review_dataset(n_samples=1800, output_path="data/reviews.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    pos_templates = [
        "Absolutely love this {item}! The quality is outstanding and delivery was super fast.",
        "Best purchase ever. Exceptional performance, highly recommend to anyone seeking {item}.",
        "Super sleek design, works like a charm. Exceeded all my expectations completely.",
        "Great customer support and high value for money. Five stars all the way!",
        "Sturdy build, reliable operation, and wonderful user interface. Very happy!"
    ]
    
    neu_templates = [
        "The {item} arrived on time. Functions as described, nothing particularly special.",
        "Average build quality. It gets the job done for the price point.",
        "Decent product overall. A few minor quirks with setup, but acceptable.",
        "Standard packaging and standard performance. Neither disappointed nor thrilled.",
        "Okay experience so far. Will update my review after a few more weeks of testing."
    ]
    
    neg_templates = [
        "Terrible experience! The {item} broke within 3 days. Complete waste of money.",
        "Extremely slow shipping and customer support was completely unhelpful. Avoid!",
        "Defective unit out of the box. Poor manufacturing quality and flimsy plastics.",
        "Very frustrated. Does not match the product description at all. Do not buy.",
        "Horrible performance and constant errors. Returning immediately for a full refund."
    ]
    
    items = ["wireless headset", "smart watch", "cloud dashboard", "laptop charger", "espresso maker", "ergonomic chair"]
    
    reviews = []
    labels = []
    
    for _ in range(n_samples):
        sentiment = np.random.choice(["positive", "neutral", "negative"], p=[0.45, 0.25, 0.30])
        item = np.random.choice(items)
        if sentiment == "positive":
            tmpl = np.random.choice(pos_templates)
            lbl = 2
        elif sentiment == "neutral":
            tmpl = np.random.choice(neu_templates)
            lbl = 1
        else:
            tmpl = np.random.choice(neg_templates)
            lbl = 0
            
        reviews.append(tmpl.format(item=item))
        labels.append(lbl)
        
    df = pd.DataFrame({"text": reviews, "label": labels, "sentiment": [["negative", "neutral", "positive"][i] for i in labels]})
    df.to_csv(output_path, index=False)
    return df
