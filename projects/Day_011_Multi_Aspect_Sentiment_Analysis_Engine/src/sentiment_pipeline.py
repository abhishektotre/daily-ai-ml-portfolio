import json
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
