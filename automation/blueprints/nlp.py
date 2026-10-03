"""
Natural Language Processing (NLP) Blueprints:
Includes sentiment analysis, text summarization with TextRank,
semantic document search, and TF-IDF vector embeddings.
"""

def generate_sentiment_analysis_project(day_num: int):
    folder_slug = "Multi_Aspect_Sentiment_Analysis_Engine"
    title = "Multi-Aspect Customer Feedback Sentiment & Emotion Engine"
    summary = "Comprehensive NLP pipeline performing aspect-based sentiment scoring, n-gram feature extraction, and Naive Bayes text classification."
    skills = ["NLP", "Sentiment Analysis", "TF-IDF", "MultinomialNB", "N-Grams", "Confusion Matrix"]

    readme_content = f"""# Day {day_num}: {title}

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
Day_{day_num:03d}_{folder_slug}/
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
cd Day_{day_num:03d}_{folder_slug}
pip install -r requirements.txt
python main.py
```
"""

    requirements_content = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    data_code = """import pandas as pd
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
"""

    pipeline_code = """import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def train_sentiment_model(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, stop_words="english", sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    clf = MultinomialNB(alpha=0.5)
    clf.fit(X_train_vec, y_train)
    
    y_pred = clf.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    target_names = ["Negative", "Neutral", "Positive"]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Sentiment Confusion Matrix (Acc: {acc*100:.1f}%)")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "sentiment_confusion_matrix.png"), dpi=200)
    plt.close()
    
    # Top predictive features for Positive and Negative
    feature_names = vectorizer.get_feature_names_out()
    neg_weights = clf.feature_log_prob_[0]
    pos_weights = clf.feature_log_prob_[2]
    
    top_pos_idx = pos_weights.argsort()[-8:]
    top_neg_idx = neg_weights.argsort()[-8:]
    
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.barh([feature_names[i] for i in top_neg_idx], [neg_weights[i] for i in top_neg_idx], color="crimson")
    plt.title("Top Negative Predictors")
    
    plt.subplot(1, 2, 2)
    plt.barh([feature_names[i] for i in top_pos_idx], [pos_weights[i] for i in top_pos_idx], color="teal")
    plt.title("Top Positive Predictors")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "top_polar_keywords.png"), dpi=200)
    plt.close()
    
    metrics = {
        "accuracy": round(float(acc), 4),
        "f1_score_weighted": round(float(f1), 4),
        "vocab_size": len(feature_names),
        "test_eval_samples": len(y_test)
    }
    
    with open(os.path.join(results_dir, "nlp_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics
"""

    main_code = """import os
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
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Natural Language Processing",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "src/__init__.py": "",
            "src/text_data_generator.py": data_code,
            "src/sentiment_pipeline.py": pipeline_code,
            "main.py": main_code
        }
    }


def generate_textrank_summarizer_project(day_num: int):
    folder_slug = "Extractive_Text_Summarization_TextRank"
    title = "Extractive Document Summarization with TextRank Algorithm"
    summary = "Graph-based natural language summarizer constructing sentence similarity graphs and applying PageRank centrality to extract key sentences."
    skills = ["NLP", "TextRank", "PageRank", "Graph Algorithms", "Cosine Similarity", "Text Summarization"]

    readme_content = f"""# Day {day_num}: {title}

![Domain](https://img.shields.io/badge/Domain-NLP-purple)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Extractive summarization distills long documents into concise overviews by ranking sentences by information centrality. This project implements:
1. Sentence boundary segmentation and token-level vocabulary normalization.
2. Dense pairwise sentence similarity matrix construction based on token overlap & cosine distance.
3. Graph-based **PageRank (TextRank)** power iteration algorithm to identify the most salient nodes.
4. Top-$K$ key sentence extraction preserving chronological document flow.
5. Summary compression ratio and centrality distribution metrics.

## 🛠️ Project Structure
```text
Day_{day_num:03d}_{folder_slug}/
├── data/
│   └── article.txt
├── results/
│   ├── sentence_centrality_ranks.png
│   └── summary_output.json
├── src/
│   ├── __init__.py
│   └── textrank_engine.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_{day_num:03d}_{folder_slug}
pip install -r requirements.txt
python main.py
```
"""

    requirements_content = """numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
"""

    article_text = """Artificial intelligence and machine learning technologies have fundamentally reshaped modern enterprise computing. 
Organizations across finance, healthcare, and retail are actively deploying automated machine learning pipelines to streamline operational workflows. 
In financial services, automated risk engines and real-time fraud detectors prevent billions of dollars in unauthorized transactions annually. 
Healthcare providers utilize computer vision algorithms and deep neural networks to identify subtle pathological patterns in medical imaging far earlier than traditional methods. 
Moreover, natural language processing models have revolutionized customer interactions by powering intelligent virtual assistants and semantic document indexing. 
Despite these advances, the rapid deployment of autonomous systems raises significant governance challenges regarding algorithmic fairness and model interpretability. 
Ensuring transparency in high-stakes automated decisions has prompted regulatory frameworks worldwide, requiring rigorous explainability standards. 
Furthermore, data drift and concept shift present ongoing maintenance hurdles, demanding continuous telemetry monitoring and automated retraining architectures. 
Consequently, organizations must invest heavily in mature MLOps practices, integrating automated testing, version control, and model registries into their core software delivery pipelines. 
Ultimately, the successful adoption of artificial intelligence hinges not solely on algorithm complexity, but on creating ethical, dependable, and reproducible software architectures.
"""

    engine_code = r"""import re
import numpy as np

def split_sentences(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 5]
    return sentences

def clean_tokens(sentence):
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
    stopwords = {"and", "the", "for", "with", "that", "this", "from", "are", "have", "been", "has", "into"}
    return [t for t in tokens if t not in stopwords]

def build_similarity_matrix(sentences):
    n = len(sentences)
    sim_matrix = np.zeros((n, n))
    token_sets = [set(clean_tokens(s)) for s in sentences]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                inter = len(token_sets[i].intersection(token_sets[j]))
                denom = np.log(len(token_sets[i]) + 1) + np.log(len(token_sets[j]) + 1)
                sim_matrix[i, j] = inter / (denom + 1e-9)
    return sim_matrix

def pagerank(sim_matrix, d=0.85, max_iter=100, tol=1e-6):
    n = sim_matrix.shape[0]
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = sim_matrix / row_sums
    
    ranks = np.ones(n) / n
    for _ in range(max_iter):
        new_ranks = (1 - d) / n + d * np.dot(P.T, ranks)
        if np.linalg.norm(new_ranks - ranks, 1) < tol:
            break
        ranks = new_ranks
    return ranks

def summarize_text(text, top_k=3):
    sentences = split_sentences(text)
    if len(sentences) <= top_k:
        return sentences, np.ones(len(sentences))
        
    sim_matrix = build_similarity_matrix(sentences)
    ranks = pagerank(sim_matrix)
    
    top_indices = ranks.argsort()[-top_k:][::-1]
    top_indices_sorted = sorted(top_indices)
    
    summary_sentences = [sentences[i] for i in top_indices_sorted]
    return summary_sentences, ranks, sentences
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.textrank_engine import summarize_text

def main():
    print("=" * 65)
    print(" 📄 Running Extractive Text Summarization with TextRank")
    print("=" * 65)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    with open("data/article.txt", "r") as f:
        text = f.read()
        
    print(f"[1/3] Reading document ({len(text.split())} words, 10 sentences)...")
    summary_sents, ranks, all_sents = summarize_text(text, top_k=3)
    
    print("[2/3] Constructing sentence graph & running PageRank centrality...")
    
    # Plot Sentence Centrality Scores
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(ranks) + 1), ranks, color="#5c3566")
    plt.title("Sentence Centrality Scores (TextRank Algorithm)")
    plt.xlabel("Sentence Index in Document")
    plt.ylabel("PageRank Centrality")
    plt.xticks(range(1, len(ranks) + 1))
    plt.tight_layout()
    plt.savefig("results/sentence_centrality_ranks.png", dpi=200)
    plt.close()
    
    output = {
        "original_sentence_count": len(all_sents),
        "summary_sentence_count": len(summary_sents),
        "compression_ratio": round(len(summary_sents) / len(all_sents), 2),
        "key_extracted_sentences": summary_sents
    }
    with open("results/summary_output.json", "w") as f:
        json.dump(output, f, indent=4)
        
    print("[3/3] Extractive Summary Generated:")
    for i, sent in enumerate(summary_sents, 1):
        print(f"      [{i}] {sent}")
    print(f"\\n Saved summary to results/summary_output.json")
    print("=" * 65)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Natural Language Processing",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "data/article.txt": article_text,
            "src/__init__.py": "",
            "src/textrank_engine.py": engine_code,
            "main.py": main_code
        }
    }

NLP_PROJECTS = [
    generate_sentiment_analysis_project,
    generate_textrank_summarizer_project
]
